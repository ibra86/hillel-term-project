### Project description

Build prediction for the data residing in
```
/data/pre_torch_dataset_enc_12m.csv 
```
having records:
```
var_obs == 0
```

### Dataset description
The input raw data represents dataset with the
following columns:
  * var_id (record id))
  * var_enum_i (i-th categories)
  * var_vec_i (i-th vector encoded vectors of vector categories)
  * var_obs (observed record indicator)
  * var_target (resulted numerical value)

If the record is observed - the resulted value is set up.

### Environmental requirements
```console
pip3 install -r requirements.txt
```

### How to run
```console
python build_prediction.py
```

After script has been completed there appear 2 resulting fils:
  * plot of the loss on train/test dataset
  ```console
  data/loss.png
  ```
  * prediction for non observed values:
  ```console
  data/dataset_predict.csv
  ```