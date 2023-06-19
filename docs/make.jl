using Documenter, MixedLRMoE

makedocs(
	doctest = false,
    modules = [MixedLRMoE],
	sitename = "MixedLRMoE.jl",
	pages = [
		"Overview" => "index.md",
        "Modelling Framework" => "framework.md",
        "Random Effects" => "random_effects.md",
		# "Expert Functions" => "experts.md",
		# "Model Initialization" => "init.md",
		"Fitting Function" => "fit.md",
		# "Predictive Functions" => "predictive.md",
		# "Examples & Tutorials" => Any[
		# 	"Data Simulation" => "examples/sim_data/simulate_data.md",
		# 	"Data Formatting" => "data_format.md",
		# 	"Fitting Function" => "examples/sim_fit/simulate_fit.md",
		# 	"Adding Customized Expert Functions" => "customize.md"
		# ],
	],
	format = Documenter.HTML(prettyurls = get(ENV, "JULIA_NO_LOCAL_PRETTY_URLS", nothing) === nothing)
)

deploydocs(
	repo = "github.com/sparktseung/MixedLRMoE.jl.git",
	branch = "gh-pages",
	devbranch = "main",
	versions = ["stable" => "v^", "v#.#.#", "dev" => "dev"],
)