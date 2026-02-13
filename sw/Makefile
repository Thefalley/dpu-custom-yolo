# DPU Software — Makefile
# Targets:
#   make sim      — Build PC simulation (printf backend)
#   make zynq     — Build for Zynq bare-metal (requires arm toolchain)
#   make clean    — Remove build artifacts

CC_SIM    = gcc
CC_ZYNQ   = arm-none-eabi-gcc
CFLAGS    = -Wall -Wextra -O2 -std=c11
SIM_FLAGS = -DPLATFORM_SIM
INCLUDES  = -Ihal -Idriver

SRCS = hal/dpu_hal.c driver/dpu_driver.c app/main.c

.PHONY: sim zynq clean

sim:
	$(CC_SIM) $(CFLAGS) $(SIM_FLAGS) $(INCLUDES) $(SRCS) -o dpu_app_sim
	@echo "Built: dpu_app_sim (PC simulation)"

zynq:
	$(CC_ZYNQ) $(CFLAGS) $(INCLUDES) $(SRCS) -o dpu_app.elf
	@echo "Built: dpu_app.elf (Zynq bare-metal)"

clean:
	rm -f dpu_app_sim dpu_app.elf
