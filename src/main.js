"use strict";

const _ = require("lodash");
const fs = require("fs");
const util = require("./utils/utils");
const tensorflow = require("./tensorflow/tensorflow");

async function main() {
  try {
    let rawdata = await fs.readFileSync("trade_history_ao_mome_trix_lstm");
    let trade_signals = JSON.parse(rawdata);

    let tensor_data = _.shuffle(util.trade_singal_extractor(trade_signals));

    console.log("Sample tensor: ", tensor_data[0]);

    await tensorflow.load_train_tensor(
      _.slice(tensor_data, 0, tensor_data.length - 200)
    );

    await tensorflow.train_modell({
      model: "lstm_hidden_cells",
      name: "",
      loop: 1
    });

    await tensorflow.test_model(_.slice(tensor_data, tensor_data.length - 200));
  } catch (e) {
    console.log(e);
  }
}

main();
