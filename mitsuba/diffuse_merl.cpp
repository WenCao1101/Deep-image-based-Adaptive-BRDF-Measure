

#include <mitsuba/core/fresolver.h>

#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#define POWITACQ_IMPLEMENTATION
#include "powitacq_rgb.h"

#define DJ_BRDF_IMPLEMENTATION 1
#include "dj_brdf.h"

#include "microfacet.h"

MTS_NAMESPACE_BEGIN


class diffuse_merl : public BSDF {
public:
	diffuse_merl(const Properties &props)
		: BSDF(props) {

		m_reflectance = new ConstantSpectrumTexture(props.getSpectrum(
			props.hasProperty("reflectance") ? "reflectance"
				: "diffuseReflectance", Spectrum(.5f)));

		fs::path m_filename = Thread::getThread()->getFileResolver()->resolve(props.getString("filename"));
    	merl_brdf = new powitacq_rgb::measured(m_filename.string().c_str());
		powitacq_rgb::Vector3f tt=merl_brdf->get_diffuse();
		//m_albedo =Color3(tt[0], tt[1], tt[2]);
		//m_albedo =Color3(0.101, 0.017, 0.01);
		m_alpha = merl_brdf->get_roughness();
	  //  m_specular = merl_brdf->get_specular();
	 //   m_specular=1.0-m_alpha;
	      m_specular=0.6f;
	    
	   m_params =djb::microfacet::params::isotropic(m_alpha);
		// load MERL
	//	m_brdf = new djb::merl(m_filename.string().c_str());
		// load tabulated
      //          m_params = djb::tabular::fit_ggx_parameters(djb::tabular(*m_brdf, 90, false));
        m_ggx = new djb::ggx();
	}

	diffuse_merl(Stream *stream, InstanceManager *manager)
		: BSDF(stream, manager) {

		configure();
	}

	~diffuse_merl()
	{
		delete merl_brdf;
                delete m_ggx;
	}

	void configure() {
		/* Verify the input parameter and fix them if necessary */
		m_components.clear();	
		m_components.push_back(EDiffuseReflection | EFrontSide | 0);
		m_components.push_back(EGlossyReflection  | EFrontSide | 0);
		m_usesRayDifferentials = false;
		BSDF::configure();
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		Color3 result(0.0f);	
       /*
     	Float phi_o=std::atan2(bRec.wo.y,bRec.wo.x);
		if(phi_o<0)
		{
			phi_o=phi_o+2*M_PI;
		}
		Float theta_o=std::acos(bRec.wo.z);
		djb::vec3 woo(std::sin(theta_o)*std::cos(phi_o),std::sin(theta_o)*std::sin(phi_o),std::cos(theta_o));

       djb::vec2 uum=m_ggx->sample_reverse(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z),woo,&m_params);*/
	   djb::vec2 uum=m_ggx->sample_reverse(djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z),djb::vec3(bRec.wo.x, bRec.wo.y, bRec.wo.z),&m_params);
	   powitacq_rgb::Vector2f uu= powitacq_rgb::Vector2f(uum.x,uum.y);
	
	   auto h = normalize(bRec.wi + bRec.wo);
	 //  auto uum=warp::uniformDiskToSquareConcentric(Point2 (bRec.wo.x,bRec.wo.y));

	//	powitacq_rgb::Vector2f uu(uum.x,uum.y);
       powitacq_rgb::Vector3f wi = powitacq_rgb::Vector3f(bRec.wi.x, bRec.wi.y, bRec.wi.z);
       powitacq_rgb::Vector3f wo = powitacq_rgb::Vector3f(bRec.wo.x, bRec.wo.y, bRec.wo.z);
        powitacq_rgb::Vector3f fr = merl_brdf->eval(wi, wo,uu);
		djb::vec3 i(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 o(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		Float inverse_j=m_ggx->inverse_jack(i, o, &m_params);
        result = Color3(fr[0], fr[1], fr[2]);
	 //   result = result*m_specular+m_albedo* INV_PI;
	      
       /*
		djb::vec3 o(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 i(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		djb::vec3 fr_p = m_brdf->evalp(i, o);
		return Color3(fr_p.x, fr_p.y, fr_p.z);*/
		return result* Frame::cosTheta(bRec.wo);
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
			|| Frame::cosTheta(bRec.wi) <= 0
			|| Frame::cosTheta(bRec.wo) <= 0)
			return 0.0f;
        Float result = 0.f;
		djb::vec3 i(bRec.wi.x, bRec.wi.y, bRec.wi.z);
		djb::vec3 o(bRec.wo.x, bRec.wo.y, bRec.wo.z);
		result= m_ggx->pdf(i, o, &m_params)*m_specular+(1-m_specular)*warp::squareToCosineHemispherePdf(bRec.wo);
        //        return m_ggx->pdf(i, o, &m_params);
		return result;
	}


	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
			return Spectrum(0.0f);
        Color3 result(0.0f);
		bool choseSpecular = true;
       if (sample.x <= m_specular) {
                choseSpecular = true;
            } else {
                choseSpecular = false;
            }

      if (choseSpecular) {

		/* Sample the tabulated microfacet BRDF */
		djb::vec3 i = djb::vec3(bRec.wi.x, bRec.wi.y, bRec.wi.z);
        djb::vec3 o = m_ggx->sample(sample.x, sample.y, i, &m_params);
       // auto o=warp::squareToCosineHemisphere(sample);
		/* Setup Mitsuba variables */
		bRec.wo = Vector(o.x, o.y, o.z);
		bRec.sampledComponent = 1;
		bRec.sampledType = EGlossyReflection;
        if (Frame::cosTheta(bRec.wo) <= 0)
			return Spectrum(0.0f);
		
	//	djb::vec3 fr_p = m_brdf->evalp(i, o) / pdf(bRec, ESolidAngle);
	}else{
		bRec.wo = warp::squareToCosineHemisphere(sample);
		bRec.sampledComponent = 0;
		bRec.sampledType = EDiffuseReflection;
	}
	    bRec.eta = 1.0f;
		
		//return Color3(fr_p.x, fr_p.y, fr_p.z);
		/* Side check */
		
        result=eval(bRec, ESolidAngle)/pdf(bRec, ESolidAngle);
		return result;	

	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf_, const Point2 &sample_) const {
		Spectrum res = sample(bRec, sample_);
		pdf_ = pdf(bRec, ESolidAngle);
		return res;
	}
   
	void addChild(const std::string &name, ConfigurableObject *child) {
		if (child->getClass()->derivesFrom(MTS_CLASS(Texture))
				&& (name == "reflectance" || name == "diffuseReflectance")) {

		} else {
			BSDF::addChild(name, child);
		}
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);

	}

	std::string toString() const {
		std::ostringstream oss;
		oss << "diffuse_merl[" << endl
			<< "  id = \"" << getID() << "\"," << endl
			<< "]";
		return oss.str();
	}

	Shader *createShader(Renderer *renderer) const;

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	Color3  m_albedo;
	Float  m_alpha;
	Float  m_specular;
	 powitacq_rgb::measured *merl_brdf;
//	djb::brdf* m_brdf;
        //djb::tabular * m_tabular;
        djb::ggx *m_ggx;
        djb::microfacet::params m_params;
};

// ================ Hardware shader implementation ================

class diffuse_merl_shader : public Shader {
public:
	diffuse_merl_shader(Renderer *renderer, const Texture *reflectance)
		: Shader(renderer, EBSDFShader), m_reflectance(reflectance) {
		m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
	}

	bool isComplete() const {
		return m_reflectanceShader.get() != NULL;
	}

	void cleanup(Renderer *renderer) {
		renderer->unregisterShaderForResource(m_reflectance.get());
	}

	void putDependencies(std::vector<Shader *> &deps) {
		deps.push_back(m_reflectanceShader.get());
	}

	void generateCode(std::ostringstream &oss,
			const std::string &evalName,
			const std::vector<std::string> &depNames) const {
		oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
			<< "    	return vec3(0.0);" << endl
			<< "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
			<< "}" << endl
			<< endl
			<< "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
			<< "    return " << evalName << "(uv, wi, wo);" << endl
			<< "}" << endl;
	}

	MTS_DECLARE_CLASS()
private:
	ref<const Texture> m_reflectance;
	ref<Shader> m_reflectanceShader;
};

Shader *diffuse_merl::createShader(Renderer *renderer) const {
	return new diffuse_merl_shader(renderer, m_reflectance.get());
}

MTS_IMPLEMENT_CLASS(diffuse_merl_shader, false, Shader)
MTS_IMPLEMENT_CLASS_S(diffuse_merl, false, BSDF)
MTS_EXPORT_PLUGIN(diffuse_merl, "diffuse_merl BRDF")
MTS_NAMESPACE_END
