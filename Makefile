_slide.so: _slide.pyx
	cython _slide.pyx
	gcc `python-config --cflags --ldflags` -fPIC -O3 -shared -o _slide.so _slide.c
