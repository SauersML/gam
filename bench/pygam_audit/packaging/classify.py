import gamfit, re, inspect, types
names=sorted(gamfit.__all__)
exc=[n for n in names if isinstance(getattr(gamfit,n,None),type) and issubclass(getattr(gamfit,n),BaseException)]
research_pat=re.compile(r'(?i)sae|dictionar|manifold|topolog|chart|atom|gumbel|ivae|rho_so|holonomy|spectrom|routab|spike|nerve|coactiv|coupling|dual_cert|e_bh|probe|log_e|bartlett|transport|crosscoder|checkpoint|separation_limit|shape_matched|shape_census|label_shuffle|gauge|equivariant|sheaf|latent|identifiab|isometry|orthogonality|sparsity|nuclear|scad|topk|threshold|ordered_beta|orderedbeta|ard|aux|mechanism|block|interp|retention|firing|census|contract|conformal|dose_response|resolution_budget|birth|fisher|variance_charge|spd|stiefel|grassmann|torus|sphere|circle|cylinder|euclidean|product|frechet|clr|alr|closure|partial_supervision|sae_supervised|whole_set|stage_contain|coordinate_posterior|recover_spikes|PeriodicHarmonic|Lie|MeasureJet|Pca|gaussian_reml|glm_reml|weighted_ridge|Penalty|PENALTY|Schedule|cuda|smoothness_penalty|duchon_function|split_likelihood|flat_block|audit|plot_atom|plot_fit|^plot$|ResponseGeometry|CtnStage|SharedPrecision|derive_|conditional_prior|basis_jet|ScoreKind|ScoreScale|BasisSpec|Descriptor|examples|kernels|diagnostics$')
research=[n for n in names if n not in exc and research_pat.search(n)]
core=[n for n in names if n not in exc and n not in research]
print('total',len(names),'exceptions',len(exc),'research-ish',len(research),'core-ish',len(core))
print('CORE:',core)
