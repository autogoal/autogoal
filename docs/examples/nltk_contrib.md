```python
from autogoal.kb import List, Sentence, CategoricalVector, ContinuousVector, MatrixContinuousDense
from autogoal.contrib import find_classes
from autogoal.kb import build_pipelines


for cls in find_classes():
    print(cls)


pipelines = build_pipelines(
    input=List(Sentence()),
    output=MatrixContinuousDense(),
    registry=find_classes()
)


for i in range(10):
    print(pipelines.sample())
```

