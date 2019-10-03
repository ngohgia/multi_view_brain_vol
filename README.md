# Visualize Brain Volumetric Data on fsaverage Surfaces

Python function to visualize volumetric data for the lateral, medial, dorsal and ventral views of both hemispheres

## Example

```python
from nilearn import image
from nilearn import datasets
from visualize_volumetric_data_on__multi_view_surf import visualize_volumetric_data_on__multi_view_surf

motor_images = datasets.fetch_neurovault_motor_task()
vol_img = motor_images.images[0]

fsaverage = datasets.fetch_surf_fsaverage()

visualize_volumetric_data_on__multi_view_surf(vol_img, fsaverage=fsaverage, threshold=1.89, title='WM_BODY')
```

Output

![Example plot](example.png)
