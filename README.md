# YetAnotherSegmentationExperiment

to fetch data:

```
from fetch.google_drive_fetcher import fetch 

fetch()
```

to generate masks:

```
from preprocessing.annotations_conversion import AnnotationsConverter
from preprocessing.config import RESOURCES_PATH, OUTPUT_DIR

converter = AnnotationsConverter(RESOURCES_PATH, OUTPUT_DIR)
converter.convert_all_images()
```