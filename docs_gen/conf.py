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
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/forestagostinelli/deepxube",
            "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                 viewBox="0 0 16 16">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53
                5.47 7.59.4.07.55-.17.55-.38
                0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49
                -2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15
                -.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72
                1.21 1.87.87 2.33.66.07-.52.28-.87.51
                -1.07-1.78-.2-3.64-.89-3.64-3.95
                0-.87.31-1.59.82-2.15-.08-.2-.36
                -1.02.08-2.12 0 0 .67-.21 2.2.82A7.65
                7.65 0 0 1 8 4.58c.68 0 1.36.09
                2 .24 1.53-1.04 2.2-.82 2.2-.82.44
                1.1.16 1.92.08 2.12.51.56.82
                1.27.82 2.15 0 3.07-1.87 3.75
                -3.65 3.95.29.25.54.73.54 1.48
                0 1.07-.01 1.93-.01 2.2 0
                .21.15.46.55.38A8.01 8.01 0 0 0
                16 8c0-4.42-3.58-8-8-8z"/>
            </svg>
        """,
            "class": "",
        },
    ],
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