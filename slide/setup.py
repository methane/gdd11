from distutils.core import setup, Extension

setup(name="slide",
      ext_modules=[Extension('_slide', ['_slide.c'])]
      )
