poetry run pylint gptanalyzer  --disable=too-many-ancestors,too-many-locals,arguments-renamed,too-many-instance-attributes,too-few-public-methods
poetry run flake8 gptanalyzer  --ignore=E203,W503