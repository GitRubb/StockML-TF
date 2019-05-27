"use strict";
const fs = require("fs");
const _ = require("lodash");

const utils = {
  trade_singal_extractor: trade_signals => {
    let tensor_datas = [];
    let tensor_good = [];
    let tensor_bad = [];
    let tensor_frame = {};

    let count_bad = 0;
    let count_good = 0;

    trade_signals.map(elem => {
      elem.map(trade => {
        // Set Buy status tensor
        tensor_frame = { input: [], output: [], proft: [] };

        // TODO use some loop this is ugly AF
        tensor_frame.input = [
          [trade.buy_in[0], trade.buy_in[1], trade.buy_in[2]],

          [trade.buy_in[3], trade.buy_in[4], trade.buy_in[5]],

          [trade.buy_in[6], trade.buy_in[7], trade.buy_in[8]],

          [trade.buy_in[9], trade.buy_in[10], trade.buy_in[11]],

          [trade.buy_in[12], trade.buy_in[13], trade.buy_in[14]]
        ];

        let buy_price = trade.buy_price[0];
        let sell_price = _.last(trade.time_history)[3];

        tensor_frame.profit = [(sell_price / trade.buy_price[0] - 1) * 100];

        if (sell_price > buy_price * 1.01) {
          tensor_frame.output = [1, 0];
          tensor_good.push(tensor_frame);
        } else {
          tensor_frame.output = [0, 1];
          tensor_bad.push(tensor_frame);
        }
      });
    });

    console.log("Good/bad", tensor_good.length, tensor_bad.length);

    // Make even dataset!
    let even_count = _.min([tensor_good.length, tensor_bad.length]);

    tensor_datas = _.take(tensor_good, even_count).concat(
      _.take(tensor_bad, even_count)
    );

    return tensor_datas;
  }
};

module.exports = utils;
