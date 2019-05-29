"use strict";
const fs = require("fs");
const _ = require("lodash");

const utils = {
  trade_singal_extractor: (trade_signals, input_indicator_counts) => {
    let result = {
      train: [],
      test: []
    };
    let tensor_good = [];
    let tensor_bad = [];
    let tensor_frame = {};

    trade_signals.map(elem => {
      elem.map(trade => {
        // Set Buy status tensor
        tensor_frame = { input: [], output: [], profit: [] };

        let shape_y = ~~(trade.buy_in.length / input_indicator_counts);
        /*
        [
        y [x,x,x],
        y [x,x,x]
        ]
        */
        tensor_frame.input = [];

        for (let i = 0; i < shape_y; i++) {
          let row = [];
          for (let k = 0; k < input_indicator_counts; k++) {
            // Add every indicator to a single row
            row.push(trade.buy_in[i * input_indicator_counts + k]);
          }
          tensor_frame.input.push(row);
        }

        let buy_price = trade.buy_price;
        // Trade history last element is the selling price
        let sell_price = trade.sell_price;

        tensor_frame.profit = [(sell_price / buy_price - 1) * 100];

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
