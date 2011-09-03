_slide.so: _slide.pyx
	cython _slide.pyx
	gcc `python-config --cflags --ldflags` -fPIC -O3 -shared -o _slide.so _slide.c

zip: _slide.pyx slidepuzzle.py
	cp _slide.pyx slidepuzzle.py source/
	zip -r source.zip source

send:
	scp _slide.c slidepuzzle.py slide.data missing p.expg:LOCAL/slide/

recv:
	rsync -avz --delete p.expg:LOCAL/slide/ p.expg/
