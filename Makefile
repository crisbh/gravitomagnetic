PYTHON ?= python
IN_DIR ?= output
IMG_DIR ?= imgs
VP_PARAMS_FILE ?= output/lcdm/parameters-usedvalues

.PHONY: test plots plot-powerspec plot-cells plot-snr plot-snr-colorbar plot-snr-colorbar-so plot-snr-colorbar-planck

test:
	$(PYTHON) -m pytest python/tests/ -v

plots: plot-powerspec plot-cells plot-snr plot-snr-colorbar

plot-powerspec:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_powerspectra.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR)

plot-cells:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_cells.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR)

plot-snr:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR)

plot-snr-colorbar: plot-snr-colorbar-so plot-snr-colorbar-planck

plot-snr-colorbar-so:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --only colorbar --colorbar-experiments SO

plot-snr-colorbar-planck:
	VP_PARAMS_FILE=$(VP_PARAMS_FILE) $(PYTHON) python/plot_snr.py --in-dir $(IN_DIR) --out-dir $(IMG_DIR) --only colorbar --colorbar-experiments Planck
