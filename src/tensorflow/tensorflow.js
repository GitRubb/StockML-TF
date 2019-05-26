"use strict";

const _ = require("lodash");
const logger = require("../logger");

const tf = require("@tensorflow/tfjs-node");
const tf_model = require("../tf_models/tf_models");

class Tensorflow {
  constructor() {
    this.train = {};
    this.predict_tensor = {};
    this.input = [];
    this.output = [];
  }

  async load_model(name) {
    try {
      this.model = await tf.loadLayersModel(`file://./${name}/model.json`);
    } catch (e) {
      logger.error("Tensorflow load_model ", e);
    }
  }

  load_train_tensor(tensor_data) {
    tensor_data.map(elem => {
      this.input.push(elem.input);
      this.output.push(elem.output);
    });
  }

  create_train_tensor() {
    this.train.input = tf.tensor(this.input);

    console.log(this.train.input.shape);

    this.train.output = tf.tensor(this.output);
  }

  create_train_tensor_volumed_smma() {
    this.train.input = tf.tensor2d(this.input);

    this.train.output = tf.tensor(this.output);
  }

  create_train_tensor_cnn() {
    this.train.input = tf
      .tensor(this.input)
      .flatten()
      .reshape([this.input.length, 48, 1]);

    this.train.output = tf.tensor(this.output);

    return;
  }

  async get_predict(input) {
    try {
      let outputs = await this.model.predict(input);

      let output_value = await outputs.dataSync();

      return output_value;
    } catch (e) {
      logger.error("Tensorflow predict error ", e);
    }
  }

  async test_model_cnn(input_tensors, output_tensors, loop = 0) {
    try {
      if (loop == 0) {
        loop = input_tensors.length;
      }

      for (let i = 0; i < loop; i++) {
        let predict_tensor = tf
          .tensor([input_tensors[i]])
          .flatten()
          .reshape([1, 48, 1]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor([output_tensors[i]]);

        let pred_res = Number(predict);
        let control_res = Number(control.dataSync());

        console.log(
          `Accurancy: ${util.round(
            (pred_res / control_res - 1) * 100
          )}%  Pred: ${util.round(pred_res * 100)} Real: ${util.round(
            control_res * 100
          )}`
        );
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async test_model(tensor_data) {
    try {
      let input_tensors = [];
      let output_tensors = [];

      tensor_data.map(elem => {
        input_tensors.push(elem.input);
        output_tensors.push(elem.output);
      });

      for (let i = 0; i < tensor_data.length; i++) {
        let predict_tensor = tf.tensor([input_tensors[i]]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor([output_tensors[i]]);

        let pred_res = Number(predict);
        let control_res = Number(control.dataSync());

        console.log(`Pred: ${predict} Real: ${control} `);
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async test_model_lstm(tensor_data) {
    try {
      let input_tensors = [];
      let output_tensors = [];

      tensor_data.map(elem => {
        input_tensors.push(elem.input);
        output_tensors.push(elem.output);
      });

      for (let i = 0; i < tensor_data.length; i++) {
        let predict_tensor = tf.tensor2d([input_tensors[i]]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor1d([output_tensors[i]]);

        let pred_res = predict;
        let control_res = control.dataSync();

        console.log(
          `Accurancy: ${_.round(
            (pred_res / control_res - 1) * 100,
            3
          )}%  Pred: ${_.round(
            pred_res * tensor_data[0].scale,
            2
          )} Real: ${_.round(
            control_res * tensor_data[0].scale,
            2
          )}  Close:  ${_.round(tensor_data[i].close, 8)}`
        );
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async test_multi_indicator(tensor_data) {
    try {
      let input_tensors = [];
      let output_tensors = [];

      tensor_data.map(elem => {
        input_tensors.push(elem.input);
        output_tensors.push(elem.output);
      });

      for (let i = 0; i < tensor_data.length; i++) {
        let predict_tensor = tf.tensor2d([input_tensors[i]]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor1d([output_tensors[i]]);

        let pred_res = predict;
        let control_res = control.dataSync();

        /*   tf_util.simple_strategy(
          pred_res * tensor_data[0].scale,
          tensor_data[i].close
        );*/

        console.log(
          `Accurancy: ${_.round(
            (pred_res / control_res - 1) * 100,
            3
          )}%  Pred: ${_.round(
            pred_res * tensor_data[0].scale,
            2
          )} Real: ${_.round(
            control_res * tensor_data[0].scale,
            2
          )}  Close:  ${_.round(tensor_data[i].close, 8)}`
        );
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async test_model_smma(tensor_data) {
    try {
      let input_tensors = [];
      let output_tensors = [];

      tensor_data.map(elem => {
        input_tensors.push(elem.input);
        output_tensors.push(elem.output);
      });

      for (let i = 0; i < tensor_data.length; i++) {
        let predict_tensor = tf.tensor1d([input_tensors[i]]);

        let predict = await this.get_predict(predict_tensor);

        let control = tf.tensor1d([output_tensors[i]]);

        let pred_res = predict;
        let control_res = control.dataSync();

        console.log(
          `Accurancy: ${util.round(
            (pred_res / control_res - 1) * 100,
            4
          )}%  Pred: ${util.round(
            pred_res * tensor_data[0].scale,
            8
          )} Real: ${util.round(
            control_res * tensor_data[0].scale,
            8
          )}  Close:  ${util.round(tensor_data[i].close, 8)}`
        );
      }
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }

  async train_modell(
    settings = {
      model: "rnn",
      name: "",
      loop: 5,
      in_shape: 1,
      out_shape: 1
    }
  ) {
    try {
      let config = {
        verbose: 1,
        shuffle: true,
        epochs: 50,
        batchSize: 4096,
        validationSplit: 0.03,
        stepsPerEpoch: 2,
        validationSteps: 2
      };

      switch (settings.model) {
        case "rnn":
          this.create_train_tensor();
          this.model = tf_model.create_model_rnn(
            settings.in_shape,
            settings.out_shape
          );
          break;
        case "lstm_v2":
          this.create_train_tensor();
          this.model = tf_model.lstm_v2(settings.in_shape, settings.out_shape);
          break;
        case "cnn":
          this.create_train_tensor_cnn();
          this.model = tf_model.create_model_cnn();
          break;
        case "lstm":
          this.create_train_tensor();
          this.model = tf_model.create_model_lstm(
            settings.in_shape,
            settings.out_shape
          );
          config = {
            verbose: 1,
            epochs: 10,
            batchSize: 4096,
            validationSplit: 0.01
          };
          break;
        default:
          this.create_train_tensor_rnn();
          this.model = tf_model.create_model_rnn(
            settings.in_shape,
            settings.out_shape
          );
          break;
      }

      let response = {};

      for (let i = 0; i < settings.loop; i++) {
        response = await this.model.fit(
          this.train.input,
          this.train.output,
          config
        );
      }

      if (settings.name != "") {
        await this.model.save(`file://./${settings.name}`);
      }

      // Return last loss this could verify if the train were success
      return _.last(response.history.loss);
    } catch (e) {
      logger.error("Tensorflow train error ", e);
    }
  }
}

module.exports = new Tensorflow();
