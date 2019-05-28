"use strict";
const fs = require("fs");
const _ = require("lodash");

const utils = {
  trade_singal_extractor: trade_signals => {
    let result = {
      train: [],
      test: []
    };
    let tensor_datas = [];
    let tensor_good = [];
    let tensor_bad = [];
    let tensor_frame = {};

    trade_signals.map(elem => {
      elem.map(trade => {
        // Set Buy status tensor
        tensor_frame = { input: [], output: [], profit: [] };

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

        if (sell_price >= buy_price * 1.01) {
          tensor_frame.output = [1, 0];
          tensor_good.push(tensor_frame);
        }

        if (sell_price <= buy_price * 0.99) {
          tensor_frame.output = [0, 1];
          tensor_bad.push(tensor_frame);
        }
      });
    });

    // Make even dataset!
    let even_count = _.min([tensor_good.length, tensor_bad.length]);

    //Mix datas
    tensor_good = _.shuffle(tensor_good);
    tensor_bad = _.shuffle(tensor_bad);

    // Create Train and Test dataset

    result.train = _.take(tensor_good, even_count).concat(
      _.take(tensor_bad, even_count)
    );

    // Shuffle Good and Bad results close to evenly
    result.train = _.shuffle(result.train);

    // First 95% for train
    result.train = _.slice(
      result.train,
      0,
      parseInt(result.train.length * 0.99)
    );

    // Last 5% for test
    result.test = _.slice(
      result.train,
      parseInt(result.train.length * 0.99),
      result.train.length
    );

    return result;
  }
};

module.exports = utils;
