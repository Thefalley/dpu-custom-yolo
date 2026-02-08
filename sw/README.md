# Software de control del DPU (sw)

Carpeta raíz del **software de control** del DPU: driver, HAL y aplicación para PC (gcc/simulación) y luego Vitis bare-metal (ZedBoard).

## Contenido

- **`docs/`** — Documentación de arquitectura (Phase W1–W6): comandos, capas, memoria, registros, secuencia de ejecución, validación.
- **`src/`** — (futuro) Código C: HAL, driver, aplicación. Se irá añadiendo según se implemente.

## Desarrollo y debug

- **Debug con `printf`:** Todo el desarrollo se hace usando `printf` (o equivalente) para trazar en consola: estados del driver, lecturas/escrituras de registros, progreso de carga, timeouts, errores. Así se puede trabajar **de forma autónoma** sin depurador: compilar, ejecutar, ver la salida y corregir.
- **En PC:** `printf` va a stdout; en Vitis bare-metal se usará `xil_printf` o UART según la BSP.
- **Niveles:** Se puede definir un nivel de verbosidad (ej. 0=silencioso, 1=resumen, 2=detalle) para no saturar la consola en producción.

## Documentación de arquitectura

- **`sw/docs/DPU_SOFTWARE_ARCHITECTURE.md`** — Resumen y enlace al documento completo.
- **Documento completo:** `../docs/DPU_SOFTWARE_ARCHITECTURE.md` (desde la raíz del repo) — Fases W1–W6: comandos, jerarquía (App / Driver / HAL / Platform), memoria, registros, secuencia de ejecución, validación.

## Cómo seguir

1. Leer `sw/docs/DPU_SOFTWARE_ARCHITECTURE.md`.
2. Implementar HAL mock (PC) con `printf` en cada acceso a “registro” o carga/lectura.
3. Implementar driver que use la HAL y haga la secuencia: load → start → wait_done → read_output.
4. Probar en PC contra mock o contra simulador RTL; luego cambiar solo la HAL para Vitis.
