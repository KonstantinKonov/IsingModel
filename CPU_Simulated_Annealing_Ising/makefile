CC = gcc
CFLAGS = -lm
LIB = libs

compile: main.c 
	gcc main.c libs/export.c libs/rand.c libs/statistics.c libs/ising.c -o main -lm

run:
	./main