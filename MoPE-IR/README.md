# MoPE-IR

Mixture-of-Prompt-Experts for Image Restoration.

Code-only snapshot of the current working project. Model architecture, routing,
and losses are retained; release edits update checkpoint defaults and remove
ordinary comments without changing executable code during comment cleanup.
This is a source-code exhibition package, not a verified turnkey reproduction.

## Layout

- `net/`: Transformer backbone, MoE prompts, and detail refinement.
- `utils/`: Dataset loaders, image processing, metrics, losses, and schedulers.
- `train.py`, `options.py`: Training entry point and arguments.
- `test.py`: Restoration evaluation and measured expert-routing collection.
- `analyze_expert_usage.py`: Existing scale-1 expert-usage analysis tool.
- `stable_diffusion/`: Bundled third-party Python sources and model configs.

No datasets, weights, generated images, experimental statistics, simulated
fixtures, simulation/data-editing scripts, browser profiles, or local caches
are included. `RELEASE_MANIFEST.json` inventories the included source files.

The release defaults to the following pretrained SD 1.5 SafeTensors checkpoint:

```text
/opt/data/private/bfs_files/PromptIR-main/ckpt/v1-5-pruned-emaonly.safetensors
```

This is a server-local path, not a download link; the weight file is not
included. Override `--diffusion_ckpt` for a different checkout. Ordinary Python
comments in the project-owned code have been removed for presentation.
Upstream attribution headers, third-party sources, and docstrings are retained.

## Before Running

Use a CUDA/PyTorch environment compatible with your original experiments and
provide your datasets and checkpoints separately. See `DEPENDENCIES.md`.
The bundled Stable Diffusion environment files are legacy upstream references,
not a validated environment lock for the modified restoration framework.

Current source limitations retained intentionally in this snapshot:

- `net/model.py` and `train.py` contain server-specific absolute paths. Adjust
  them for your checkout before running, including the diffusion YAML path.
- `net/model.py` imports `sd3_impls` from an external `sd3.5-main` directory.
  That directory is not present in this workspace or archive. Resolve that
  dependency, or review the unused import, before importing the model.
- Dataset indexes expected under `data_dir/`, image datasets, restoration
  checkpoints, and the pretrained diffusion checkpoint are not distributed.
- Training currently defaults to `dehaze` and the model defaults to three
  experts, whereas testing defaults to five. Match task settings, expert count,
  and checkpoint shapes explicitly; defaults are not a paper configuration.
- Evaluation uses random latent perturbation in the current implementation.
  Account for inference randomness when reporting comparisons.

GPU execution, dependency installation, and end-to-end reproducibility have
not been validated as part of packaging. The archive is checked for file
integrity and exclusion of non-code artifacts only.

## Attribution And Licensing

The model retains the PromptIR attribution in `net/model.py`. Third-party
Stable Diffusion and taming source licenses/notices are preserved alongside
their code. Review these terms and the upstream PromptIR license before public
distribution. No new blanket license is assigned to the combined project.
Add your paper title, authors, citation, verified environment, and reproduction
instructions before presenting this as an official reproducibility release.
