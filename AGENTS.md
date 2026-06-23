# Engineering Rules

- Do not modify anything under `third_party/`.
- Target Python 3.14 or later for development and test execution.
- Do not silently change random seeds. Any seed-policy change must be explicit in configuration and documentation.
- Every generation parameter must be recorded in the configuration and in the manifest.
- Do not generate the complete 800-instance dataset unless explicitly requested.
- Run `pytest` after modifying code.
- Do not commit uncompressed LP (`.lp`) files. Dataset artifacts must use compressed `.lp.gz` files.
