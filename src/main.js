"use strict";

const _ = require("lodash");
const fs = require("fs");
const util = require("./utils/utils");
const tensorflow = require("./tensorflow/tensorflow");

async function main() {
  try {
    let rawdata = await fs.readFileSync("trade_history_ao_mome_trix_lstm");
    let trade_signals = JSON.parse(rawdata);

    let tensor_data = util.trade_singal_extractor(trade_signals);

    console.log("Train sample data: ", tensor_data.train[0]);

    await tensorflow.load_train_tensor(tensor_data.train);

    await tensorflow.train_modell({
      model: "lstm_hidden_cells",
      name: "",
      loop: 1,
      epochs: 20
    });

    /* await tensorflow.re_train({
      loop: 1,
      epochs: 1
    });*/

    //await tensorflow.load_model("lstm_test_1");

    await tensorflow.test_model(tensor_data.test);

    console.log("Test sample data: ", tensor_data.train[0]);
  } catch (e) {
    console.log(e);
  }
}

main();
