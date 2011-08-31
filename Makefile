slide: slide.pyx
	cython --embed slide.pyx
	gcc -O3 `python-config --cflags --libs` -o slide slide.c

run: slide
	./slide
