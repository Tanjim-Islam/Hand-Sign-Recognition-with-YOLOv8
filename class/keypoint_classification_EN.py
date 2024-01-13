{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 185,
      "metadata": {
        "id": "igMyGnjE9hEp"
      },
      "outputs": [],
      "source": [
        "import csv\n",
        "\n",
        "import numpy as np\n",
        "import tensorflow as tf\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "RANDOM_SEED = 42"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 186,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Num GPUs Available:  0\n"
          ]
        }
      ],
      "source": [
        "print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t2HDvhIu9hEr"
      },
      "source": [
        "# Specify each path"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 187,
      "metadata": {
        "id": "9NvZP2Zn9hEy"
      },
      "outputs": [],
      "source": [
        "dataset = 'model/keypoint_classifier/keypoint.csv'\n",
        "model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'\n",
        "tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "s5oMH7x19hEz"
      },
      "source": [
        "# Set number of classes"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 188,
      "metadata": {
        "id": "du4kodXL9hEz"
      },
      "outputs": [],
      "source": [
        "NUM_CLASSES = 8"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XjnL0uso9hEz"
      },
      "source": [
        "# Dataset reading"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 189,
      "metadata": {
        "id": "QT5ZqtEz9hE0"
      },
      "outputs": [],
      "source": [
        "X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 190,
      "metadata": {
        "id": "QmoKFsp49hE0"
      },
      "outputs": [],
      "source": [
        "y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 191,
      "metadata": {
        "id": "xQU7JTZ_9hE0"
      },
      "outputs": [],
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mxK_lETT9hE0"
      },
      "source": [
        "# Model building"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 192,
      "metadata": {
        "id": "vHBmUf1t9hE1"
      },
      "outputs": [],
      "source": [
        "model = tf.keras.models.Sequential([\n",
        "    tf.keras.layers.Input((21 * 2, )),\n",
        "    tf.keras.layers.Dropout(0.2),\n",
        "    tf.keras.layers.Dense(20, activation='relu'),\n",
        "    tf.keras.layers.Dropout(0.4),\n",
        "    tf.keras.layers.Dense(10, activation='relu'),\n",
        "    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')\n",
        "])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 193,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ypqky9tc9hE1",
        "outputId": "5db082bb-30e3-4110-bf63-a1ee777ecd46"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Model: \"sequential_8\"\n",
            "_________________________________________________________________\n",
            " Layer (type)                Output Shape              Param #   \n",
            "=================================================================\n",
            " dropout_16 (Dropout)        (None, 42)                0         \n",
            "                                                                 \n",
            " dense_24 (Dense)            (None, 20)                860       \n",
            "                                                                 \n",
            " dropout_17 (Dropout)        (None, 20)                0         \n",
            "                                                                 \n",
            " dense_25 (Dense)            (None, 10)                210       \n",
            "                                                                 \n",
            " dense_26 (Dense)            (None, 8)                 88        \n",
            "                                                                 \n",
            "=================================================================\n",
            "Total params: 1158 (4.52 KB)\n",
            "Trainable params: 1158 (4.52 KB)\n",
            "Non-trainable params: 0 (0.00 Byte)\n",
            "_________________________________________________________________\n"
          ]
        }
      ],
      "source": [
        "model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 194,
      "metadata": {
        "id": "MbMjOflQ9hE1"
      },
      "outputs": [],
      "source": [
        "# Model checkpoint callback\n",
        "cp_callback = tf.keras.callbacks.ModelCheckpoint(\n",
        "    model_save_path, verbose=1, save_weights_only=False)\n",
        "# Callback for early stopping\n",
        "es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 195,
      "metadata": {
        "id": "c3Dac0M_9hE2"
      },
      "outputs": [],
      "source": [
        "# Model compilation\n",
        "model.compile(\n",
        "    optimizer='adam',\n",
        "    loss='sparse_categorical_crossentropy',\n",
        "    metrics=['accuracy']\n",
        ")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7XI0j1Iu9hE2"
      },
      "source": [
        "# Model training"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 196,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "WirBl-JE9hE3",
        "outputId": "71b30ca2-8294-4d9d-8aa2-800d90d399de",
        "scrolled": true
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/500\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "1/3 [=========>....................] - ETA: 2s - loss: 2.1867 - accuracy: 0.0938\n",
            "Epoch 1: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "e:\\asl\\env\\Lib\\site-packages\\keras\\src\\engine\\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 488/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6205 - accuracy: 0.7656\n",
            "Epoch 488: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 209ms/step - loss: 0.6157 - accuracy: 0.7719 - val_loss: 0.3734 - val_accuracy: 0.9158\n",
            "Epoch 489/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6572 - accuracy: 0.7656\n",
            "Epoch 489: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 43ms/step - loss: 0.5868 - accuracy: 0.7719 - val_loss: 0.3733 - val_accuracy: 0.9158\n",
            "Epoch 490/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6891 - accuracy: 0.7656\n",
            "Epoch 490: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 35ms/step - loss: 0.6369 - accuracy: 0.7719 - val_loss: 0.3736 - val_accuracy: 0.9158\n",
            "Epoch 491/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6733 - accuracy: 0.7422\n",
            "Epoch 491: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 32ms/step - loss: 0.6742 - accuracy: 0.7298 - val_loss: 0.3738 - val_accuracy: 0.9158\n",
            "Epoch 492/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6363 - accuracy: 0.7812\n",
            "Epoch 492: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 39ms/step - loss: 0.6616 - accuracy: 0.7439 - val_loss: 0.3739 - val_accuracy: 0.9158\n",
            "Epoch 493/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.5963 - accuracy: 0.7969\n",
            "Epoch 493: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 41ms/step - loss: 0.5806 - accuracy: 0.7825 - val_loss: 0.3739 - val_accuracy: 0.9158\n",
            "Epoch 494/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.7567 - accuracy: 0.7500\n",
            "Epoch 494: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 42ms/step - loss: 0.6373 - accuracy: 0.7754 - val_loss: 0.3741 - val_accuracy: 0.9158\n",
            "Epoch 495/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6155 - accuracy: 0.7812\n",
            "Epoch 495: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 40ms/step - loss: 0.5901 - accuracy: 0.7860 - val_loss: 0.3742 - val_accuracy: 0.9158\n",
            "Epoch 496/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.7220 - accuracy: 0.6953\n",
            "Epoch 496: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 32ms/step - loss: 0.7009 - accuracy: 0.7263 - val_loss: 0.3741 - val_accuracy: 0.9158\n",
            "Epoch 497/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6782 - accuracy: 0.7734\n",
            "Epoch 497: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 33ms/step - loss: 0.6716 - accuracy: 0.7614 - val_loss: 0.3731 - val_accuracy: 0.9158\n",
            "Epoch 498/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6959 - accuracy: 0.7578\n",
            "Epoch 498: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 38ms/step - loss: 0.6720 - accuracy: 0.7474 - val_loss: 0.3705 - val_accuracy: 0.9158\n",
            "Epoch 499/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.5563 - accuracy: 0.8203\n",
            "Epoch 499: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 34ms/step - loss: 0.5852 - accuracy: 0.7825 - val_loss: 0.3691 - val_accuracy: 0.9158\n",
            "Epoch 500/500\n",
            "1/3 [=========>....................] - ETA: 0s - loss: 0.6695 - accuracy: 0.7500\n",
            "Epoch 500: saving model to model/keypoint_classifier\\keypoint_classifier.hdf5\n",
            "3/3 [==============================] - 0s 63ms/step - loss: 0.6182 - accuracy: 0.7649 - val_loss: 0.3683 - val_accuracy: 0.9158\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.src.callbacks.History at 0x274d35567d0>"
            ]
          },
          "execution_count": 196,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "model.fit(\n",
        "    X_train,\n",
        "    y_train,\n",
        "    epochs=500,\n",
        "    batch_size=128,\n",
        "    validation_data=(X_test, y_test),\n",
        "    callbacks=[cp_callback, es_callback]\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 197,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pxvb2Y299hE3",
        "outputId": "59eb3185-2e37-4b9e-bc9d-ab1b8ac29b7f"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "1/1 [==============================] - 0s 54ms/step - loss: 0.3683 - accuracy: 0.9158\n"
          ]
        }
      ],
      "source": [
        "# Model evaluation\n",
        "val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 198,
      "metadata": {
        "id": "RBkmDeUW9hE4"
      },
      "outputs": [],
      "source": [
        "# Loading the saved model\n",
        "model = tf.keras.models.load_model(model_save_path)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 199,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tFz9Tb0I9hE4",
        "outputId": "1c3b3528-54ae-4ee2-ab04-77429211cbef"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "1/1 [==============================] - 0s 81ms/step\n",
            "[4.1953423e-03 1.4251891e-02 2.4385352e-03 2.4946049e-02 6.5626595e-03\n",
            " 2.1649878e-01 7.3040241e-01 7.0434524e-04]\n",
            "6\n"
          ]
        }
      ],
      "source": [
        "# Inference test\n",
        "predict_result = model.predict(np.array([X_test[0]]))\n",
        "print(np.squeeze(predict_result))\n",
        "print(np.argmax(np.squeeze(predict_result)))"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "S3U4yNWx9hE4"
      },
      "source": [
        "# Confusion matrix"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 200,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 582
        },
        "id": "AP1V6SCk9hE5",
        "outputId": "08e41a80-7a4a-4619-8125-ecc371368d19"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "3/3 [==============================] - 0s 2ms/step\n"
          ]
        },
        {
          "data": {
            "image/png": "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",
            "text/plain": [
              "<Figure size 700x600 with 2 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Classification Report\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00        14\n",
            "           1       1.00      1.00      1.00        14\n",
            "           2       0.72      1.00      0.84        13\n",
            "           3       1.00      1.00      1.00         5\n",
            "           4       1.00      1.00      1.00         4\n",
            "           5       1.00      0.33      0.50        12\n",
            "           6       0.86      1.00      0.92        18\n",
            "           7       1.00      1.00      1.00        15\n",
            "\n",
            "    accuracy                           0.92        95\n",
            "   macro avg       0.95      0.92      0.91        95\n",
            "weighted avg       0.93      0.92      0.90        95\n",
            "\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.metrics import confusion_matrix, classification_report\n",
        "\n",
        "def print_confusion_matrix(y_true, y_pred, report=True):\n",
        "    labels = sorted(list(set(y_true)))\n",
        "    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)\n",
        "    \n",
        "    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)\n",
        " \n",
        "    fig, ax = plt.subplots(figsize=(7, 6))\n",
        "    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)\n",
        "    ax.set_ylim(len(set(y_true)), 0)\n",
        "    plt.show()\n",
        "    \n",
        "    if report:\n",
        "        print('Classification Report')\n",
        "        print(classification_report(y_test, y_pred))\n",
        "\n",
        "Y_pred = model.predict(X_test)\n",
        "y_pred = np.argmax(Y_pred, axis=1)\n",
        "\n",
        "print_confusion_matrix(y_test, y_pred)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FNP6aqzc9hE5"
      },
      "source": [
        "# Convert to model for Tensorflow-Lite"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 201,
      "metadata": {
        "id": "ODjnYyld9hE6"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "e:\\asl\\env\\Lib\\site-packages\\keras\\src\\engine\\training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
            "  saving_api.save_model(\n"
          ]
        }
      ],
      "source": [
        "# Save as a model dedicated to inference\n",
        "model.save(model_save_path, include_optimizer=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 202,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zRfuK8Y59hE6",
        "outputId": "a4ca585c-b5d5-4244-8291-8674063209bb"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "INFO:tensorflow:Assets written to: C:\\Users\\User\\AppData\\Local\\Temp\\tmpkx64l9ya\\assets\n"
          ]
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "INFO:tensorflow:Assets written to: C:\\Users\\User\\AppData\\Local\\Temp\\tmpkx64l9ya\\assets\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "6784"
            ]
          },
          "execution_count": 202,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Transform model (quantization)\n",
        "\n",
        "converter = tf.lite.TFLiteConverter.from_keras_model(model)\n",
        "converter.optimizations = [tf.lite.Optimize.DEFAULT]\n",
        "tflite_quantized_model = converter.convert()\n",
        "\n",
        "open(tflite_save_path, 'wb').write(tflite_quantized_model)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CHBPBXdx9hE6"
      },
      "source": [
        "# Inference test"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 203,
      "metadata": {
        "id": "mGAzLocO9hE7"
      },
      "outputs": [],
      "source": [
        "interpreter = tf.lite.Interpreter(model_path=tflite_save_path)\n",
        "interpreter.allocate_tensors()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 204,
      "metadata": {
        "id": "oQuDK8YS9hE7"
      },
      "outputs": [],
      "source": [
        "# Get I / O tensor\n",
        "input_details = interpreter.get_input_details()\n",
        "output_details = interpreter.get_output_details()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 205,
      "metadata": {
        "id": "2_ixAf_l9hE7"
      },
      "outputs": [],
      "source": [
        "interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 206,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "s4FoAnuc9hE7",
        "outputId": "91f18257-8d8b-4ef3-c558-e9b5f94fabbf",
        "scrolled": true
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "CPU times: total: 0 ns\n",
            "Wall time: 0 ns\n"
          ]
        }
      ],
      "source": [
        "%%time\n",
        "# Inference implementation\n",
        "interpreter.invoke()\n",
        "tflite_results = interpreter.get_tensor(output_details[0]['index'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 207,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vONjp19J9hE8",
        "outputId": "77205e24-fd00-42c4-f7b6-e06e527c2cba"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "[4.19534277e-03 1.42518915e-02 2.43853522e-03 2.49460507e-02\n",
            " 6.56265905e-03 2.16498762e-01 7.30402470e-01 7.04345643e-04]\n",
            "6\n"
          ]
        }
      ],
      "source": [
        "print(np.squeeze(tflite_results))\n",
        "print(np.argmax(np.squeeze(tflite_results)))"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "collapsed_sections": [],
      "name": "keypoint_classification_EN.ipynb",
      "provenance": [],
      "toc_visible": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
