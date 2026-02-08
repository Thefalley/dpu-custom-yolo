# DPU Control Software Architecture

**Documento principal:** La especificación completa (Phase W1–W6) está en la raíz del proyecto:

- **`../../docs/DPU_SOFTWARE_ARCHITECTURE.md`**

Este archivo es una referencia. Para trabajar desde la carpeta `sw/`, usa el doc de la raíz o copia su contenido aquí si quieres que `sw/` sea totalmente autocontenido.

## Resumen rápido

- **Phase W1:** Comandos del DPU (RESET, CONFIGURE, LOAD_INPUT, LOAD_WEIGHTS, START, STOP, STATUS).
- **Phase W2:** Jerarquía App → Driver → HAL → Platform; solo HAL cambia entre PC y Vitis.
- **Phase W3:** Formato imagen INT8, mapa de memoria lineal, flujo de carga (PIO/DMA).
- **Phase W4:** Mapa de registros (CONTROL, STATUS, ADDR, DATA_WR, DATA_RD).
- **Phase W5:** Secuencia de ejecución y pseudo-código; manejo de errores.
- **Phase W6:** Validación en PC (mock/sim) y reuso en Vitis.

Ver el documento completo en `docs/DPU_SOFTWARE_ARCHITECTURE.md` (desde la raíz del repo).
