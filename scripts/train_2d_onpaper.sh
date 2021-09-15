#! /bin/bash
export PATH_POSTFIX=$1
export MISC_ARGS=$2

export DATA_ROOT="./outputs/2d"
export TRAINER=${TRAINER:-ImageCorrespondenceTrainer}
export INLIER_MODEL=${INLIER_MODEL:-ResNetSC}
export DATASET=${DATASET:-YFCC100MDatasetExtracted}
export OPTIMIZER=${OPTIMIZER:-SGD}
export LR=${LR:-1e-1}
export MAX_EPOCH=${MAX_EPOCH:-100}
export BATCH_SIZE=${BATCH_SIZE:-32}
export QUANTIZATION_SIZE=${QUANTIZATION_SIZE:-0.01}
export INLIER_THRESHOLD_PIXEL=${INLIER_THRESHOLD_PIXEL:-0.01}
export INLIER_FEATURE_TYPE=${INLIER_FEATURE_TYPE:-coords}
export COLLATION_2D=${COLLATION_2D:-collate_correspondence}
export BEST_VAL_METRIC=${BEST_VAL_METRIC:-ap}
export TIME=$(date +"%Y-%m-%d_%H-%M-%S")
export VERSION=$(git rev-parse HEAD)

export OUT_DIR=${DATA_ROOT}/${DATASET}-v${QUANTIZATION_SIZE}-i${INLIER_THRESHOLD_PIXEL}/${INLIER_MODEL}-${BEST_VAL_METRIC}-${INLIER_FEATURE_TYPE}/${OPTIMIZER}-lr${LR}-e${MAX_EPOCH}-b${BATCH_SIZE}/${PATH_POSTFIX}/${TIME}

export PYTHONUNBUFFERED="True"

echo $OUT_DIR

mkdir -m 755 -p $OUT_DIR

LOG=${OUT_DIR}/log_${TIME}.txt

echo "Host: " $(hostname) | tee -a $LOG
echo "Conda " $(which conda) | tee -a $LOG
echo $(pwd) | tee -a $LOG
echo "Version: " $VERSION | tee -a $LOG
echo "Git diff" | tee -a $LOG
echo "" | tee -a $LOG
git diff | tee -a $LOG
echo "" | tee -a $LOG
nvidia-smi | tee -a $LOG

# Training
python train.py \
	--optimizer ${OPTIMIZER} \
	--lr ${LR} \
	--batch_size ${BATCH_SIZE} \
	--val_batch_size ${BATCH_SIZE} \
	--max_epoch ${MAX_EPOCH} \
	--dataset ${DATASET} \
	--trainer ${TRAINER} \
	--inlier_model ${INLIER_MODEL} \
	--inlier_feature_type ${INLIER_FEATURE_TYPE} \
	--quantization_size ${QUANTIZATION_SIZE} \
	--inlier_threshold_pixel ${INLIER_THRESHOLD_PIXEL} \
	--collation_2d ${COLLATION_2D} \
	--best_val_metric ${BEST_VAL_METRIC} \
	--out_dir ${OUT_DIR} \
	--sample_minimum_coords True \
	${MISC_ARGS} 2>&1 | tee -a $LOG

# Test
# TODO
