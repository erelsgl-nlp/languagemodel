var util = require("util");
var logSumExp = require('./logSumExp');


/**
 * This class represents a simple, unigram-based language model.
 * Based on:
 *
 * Leuski Anton, Traum David. A Statistical Approach for Text Processing in Virtual Humans tech. rep.University of Southern California, Institute for Creative Technologies 2008.
 * http://www.citeulike.org/user/erelsegal-halevi/article/12540655
 * 
 * @author Erel Segal-Halevi
 * @since 2013-08
 * 
 * opts - may contain the following options:
 * * smoothingCoefficient - the lambda-factor for smoothing the unigram probabilities.
 */
var LanguageModel = function(opts) {
	this.smoothingCoefficient = opts.smoothingCoefficient || 0.9;
}

LanguageModel.prototype = {

	/**
	 * Train the language with all the given documents.
	 * 
	 * @param dataset
	 *      an array with hashes of the format: 
	 *            {word1:count1, word2:count2,...}
	 *      each object represents the a sentence (it should be tokenized in advance). 
	 */
	trainBatch : function(dataset) {

		// calculate counts for equation (3):
		var mapWordToTotalCount = {};
		var totalNumberOfWordsInDataset = 0;
		for (var i in dataset) {
			var datum = dataset[i];
			var totalPerDatum = 0;
			
			// for each input sentence, count the total number of words in it:
			for (var word in datum) {
				mapWordToTotalCount[word] |= 0;
				mapWordToTotalCount[word] += datum[word];
				totalPerDatum += datum[word];
			}
			datum["_total"] = totalPerDatum;
			totalNumberOfWordsInDataset += totalPerDatum;
		}
		mapWordToTotalCount["_total"] = totalNumberOfWordsInDataset;
		
		this.dataset = dataset;
		this.mapWordToTotalCount = mapWordToTotalCount;
		
		// calculate smoothing factor for equation (3):
		var mapWordToSmoothingFactor = {};
		for (var word in mapWordToTotalCount) {
			mapWordToSmoothingFactor[word] = 
				(1-this.smoothingCoefficient) * this.mapWordToTotalCount[word] / this.mapWordToTotalCount["_total"];
		}
		this.mapWordToSmoothingFactor = mapWordToSmoothingFactor;
	},
	
	/**
	 * @return the map of all words in the training Dataset, each word with its total count in the Dataset.
	 */
	getAllWordCounts: function() {
		return this.mapWordToTotalCount;
	},
	
	/**
	 * @param sentenceCounts a hash {word1: count1, word2: count2, ... "_total": totalCount}, representing a sentence.
	 * @return the log-probability of that sentence, given the model built from the Dataset.
	 */
	logProbSentenceGivenDataset: function(sentenceCounts) {  // (2) log P(w1...wn) = ...
		var logProducts = [];
		for (var i in this.dataset) {
			var datum = this.dataset[i];
			logProducts.push(this.logProbSentenceGivenSentence(sentenceCounts, datum));
		}
		var logSentenceLikelihood = logSumExp(logProducts);
		
		return logSentenceLikelihood - Math.log(this.dataset.length);
	},

	/**
	 * @param sentenceCounts a hash {word1: count1, word2: count2, ... "_total": totalCount}, representing a sentence.
	 * @param givenSentenceCounts a hash {word1: count1, word2: count2, ... "_total": totalCount}, representing a sentence.
	 * @return the (smoothed) log product probabilities that the words in sentenceCounts appear in the givenSentenceCounts.
	 */
	logProbSentenceGivenSentence: function(sentenceCounts, givenSentenceCounts) {
		var logProduct=0;
		for (var word in sentenceCounts)
			logProduct += sentenceCounts[word] * this.logProbWordGivenSentence(word, givenSentenceCounts);
		return logProduct;
	},
	
	/**
	 * @param word a word from the INPUT domain.
	 * @param givenSentenceCounts a hash {word1: count1, word2: count2, ... "_total": totalCount}, representing a sentence.
	 * @return the (smoothed) probability that the word appears in the sentence.
	 */
	logProbWordGivenSentence: function(word, givenSentenceCounts) {  // (3) p_s(w) =~ pi_s(w) = ...
		var totalGivenSentenceCounts = ("_total" in givenSentenceCounts?
				givenSentenceCounts["_total"]:
				Object.keys(givenSentenceCounts).
				    map(function(key){return givenSentenceCounts[key]}).
				    reduce(function(memo, num){ return memo + num; }, 0));

		var prob = (word in givenSentenceCounts? 
				this.smoothingCoefficient     * givenSentenceCounts[word] / totalGivenSentenceCounts +      this.mapWordToSmoothingFactor[word]:
				this.mapWordToSmoothingFactor[word]);
		if (isNaN(prob)) {
			console.log(util.inspect(this,{depth:3}));
			throw new Error("logProbWordGivenSentence("+word+", "+JSON.stringify(givenSentenceCounts)+") is NaN!");
		}
		return Math.log(prob);
	},
}




/*
 * UTILITY FUNCTIONS
 */

module.exports = LanguageModel;



if (process.argv[1] === __filename) {
	console.log("LanguageModel.js demo start");
	
	var model = new LanguageModel({
		smoothingFactor : 0.9,
	});

	model.trainBatch([
		{"I":1,"offer":1,"a":1,"salary":1,"of":1,"20000":1},
		{"I":1,"offer":1,"a":1,"salary":1,"of":1,"7000":1},
		{"I":1,"offer":1,"a":2,"car":1,"and":1,"pension":1},
	    ]);

	console.log(model.logProbSentenceGivenDataset(
			{"I":1,"offer":1,"a":1,"salary":1,"of":1,"20000":1,"and":1,"car":1}
			));                  

	
	console.log("LanguageModel.js demo end");
}
