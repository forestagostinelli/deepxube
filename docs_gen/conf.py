# docs_tutorial/conf.py

project = "DeepXube"
author = "Forest Agostinelli"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
    "autodoc2",
]


autodoc2_packages = [
    {
        "path": "../deepxube",
        "module": "deepxube",
        "exclude_files": [],
    }
]

autodoc2_output_dir = "apidocs"
autodoc2_render_plugin = "rst"
autodoc2_module_summary = True

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

html_theme = "furo"
# html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0f172a",
        "color-brand-content": "#0f172a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#60a5fa",
        "color-brand-content": "#e6eefc",
    },
    "source_repository": "https://github.com/forestagostinelli/deepxube",
    "source_branch": "main",
    "source_directory": "docs_gen/",
    "top_of_page_buttons": ["view"],
}

html_context = {
    "github_user": "forestagostinelli",
    "github_repo": "deepxube",
    "github_version": "main",
    "doc_path": "docs_gen/",
}

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]

bibtex_bibfiles = ["references.bib"]