"""Put copies of Colab's `/content/sample_data` files in a local cache.

    python examples/colab/fetch_sample_data.py

Every Colab runtime has a `sample_data` folder. Outside Colab, the same files come
from their original sources:

* california_housing_train.csv and california_housing_test.csv: Google's Machine
  Learning Crash Course (download.mlcc.google.com), the files Colab ships;
* mnist_train_small.csv (the first 20,000 training images) and mnist_test.csv (all
  10,000 test images): MNIST from the mirror torchvision downloads from, written in
  Colab's format (label, then 784 pixel values, no header).

The cache is ~/.cache/silent_bug_agent/sample_data, or SBA_SAMPLE_DATA if set.
"""

from __future__ import annotations

import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from variants import has_sample_data, sample_data_dir  # noqa: E402

HOUSING = "https://download.mlcc.google.com/mledu-datasets/{name}"


def main() -> int:
    folder = sample_data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("california_housing_train.csv", "california_housing_test.csv"):
        if not (folder / name).is_file():
            print(f"downloading {name}")
            urllib.request.urlretrieve(HOUSING.format(name=name), folder / name)
    if not (folder / "mnist_train_small.csv").is_file() or not (folder / "mnist_test.csv").is_file():
        import numpy as np
        import torchvision

        with tempfile.TemporaryDirectory() as raw:
            for train, rows, name in ((True, 20000, "mnist_train_small.csv"), (False, 10000, "mnist_test.csv")):
                print(f"writing {name}")
                data = torchvision.datasets.MNIST(raw, train=train, download=True)
                table = np.concatenate([data.targets[:rows].numpy()[:, None],
                                        data.data[:rows].reshape(rows, -1).numpy()], axis=1)
                np.savetxt(folder / name, table, fmt="%d", delimiter=",")
    print(f"sample data in {folder}: {'complete' if has_sample_data(folder) else 'incomplete'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
