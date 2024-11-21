DATA_DIR="./data/robomimic"

wget http://downloads.cs.stanford.edu/downloads/rt_benchmark/lift/ph/demo_v141.hdf5 -P $DATA_DIR/


# lift - ph
python external/robomimic/robomimic/scripts/dataset_states_to_obs.py --done_mode 2 \
--dataset $DATA_DIR/demo_v141.hdf5 \
--output_name low_dim_v141.hdf5
python external/robomimic/robomimic/scripts/dataset_states_to_obs.py --done_mode 2 \
--dataset $DATA_DIR/demo_v141.hdf5 \
--output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84


python external/diffusion_policy/diffusion_policy/scripts/robomimic_dataset_conversion.py -i $DATA_DIR/low_dim_v141.hdf5 -o $DATA_DIR/low_dim_abs_v141.hdf5
python external/diffusion_policy/diffusion_policy/scripts/robomimic_dataset_conversion.py -i $DATA_DIR/image_v141.hdf5 -o $DATA_DIR/image_abs_v141.hdf5
