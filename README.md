Experiments in generating interesting Voronoi decompositions using Python.


# Running a program


Current directory = `~/dev/generative/voronoi`

Execute something like this:

```
python python/mosaic_from_image.py images_in/kandinsky-1908-1.png 256000 1
```

where

```
python path-to-mosaic_from_image.py path-to-input-png-file number-of-points-in-voronoi-diagram chunk-size
```


```
$ source env/bin/activate
$ python python/python/voronoi4g.py
$ open voronoi_color_4g.svg
$ deactivate
```

# Setting up

```
python -m venv myenv
source myenv/bin/activate
pip install PACKAGE
```
# Managing

```
pip list
pip freeze >requirements.txt
pip install -r requirements.txt
```

# Dependencies

```
contourpy==1.2.1
cycler==0.12.1
fonttools==4.53.1
kiwisolver==1.4.5
matplotlib==3.9.1
numpy==2.0.1
packaging==24.1
pillow==10.4.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
scipy==1.14.0
six==1.16.0
svgwrite==1.4.3
```

