#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

VERSION=${VERSION:-las1}
MODEL_SIZE=${MODEL_SIZE:-m}

if [ -z "${CKPT}" ]; then
  case "${VERSION}" in
    las2|v2|2|liteanystereov2)
      case "${MODEL_SIZE}" in
        s|S)
          CKPT=./checkpoints/LAS2_S.pth
          ;;
        m|M)
          CKPT=./checkpoints/LAS2_M.pth
          ;;
        l|L)
          CKPT=./checkpoints/LAS2_L.pth
          ;;
        h|H)
          CKPT=./checkpoints/LAS2_H.pth
          ;;
        *)
          echo "Unknown LAS2 MODEL_SIZE=${MODEL_SIZE}; expected s, m, l, or h" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      CKPT=./checkpoints/LiteAnyStereo.pth
      ;;
  esac
fi

python evaluate_stereo.py --version "${VERSION}" --model_size "${MODEL_SIZE}" --restore_ckpt "${CKPT}" --dataset middlebury_H
python evaluate_stereo.py --version "${VERSION}" --model_size "${MODEL_SIZE}" --restore_ckpt "${CKPT}" --dataset eth3d
python evaluate_stereo.py --version "${VERSION}" --model_size "${MODEL_SIZE}" --restore_ckpt "${CKPT}" --dataset kitti
python evaluate_stereo.py --version "${VERSION}" --model_size "${MODEL_SIZE}" --restore_ckpt "${CKPT}" --dataset drivingstereo
