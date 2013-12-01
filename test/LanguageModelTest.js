/**
 * a unit-test for utils related to Homer classifier
 * 
 * @author Erel Segal-Halevi
 * @since 2013-08
 */

var LanguageModel = require('../LanguageModel');
var wordcounts = require('../wordcounts');
var random = require('./generaterandom');
var should = require('should');

describe('Language Model', function() {
	
	var model = new LanguageModel({
		smoothingCoefficient : 0.9,
	});
	
	var assertProbSentence = function(sentence, expected) {
		var p = Math.exp(model.logProbSentenceGivenDataset(wordcounts(sentence)));
		if (Math.abs(p-expected)/expected>0.01) 
			throw new Error("p("+sentence+") = "+p+" should be "+expected);
	}

	random_string = random.random_string(15)
	
	it('Probabilities of all words of the sentence given the sentence converges to 2 ', function() {	
		str = random_string
		
		model.smoothingCoefficient = 1
		model.trainBatch([wordcounts(str)])

		str = str.split(" ")

		var prob = 0
		for (word in str)
		{
			prob += Math.exp(model.logProbWordGivenSentence(str[word], wordcounts(str.join(" "))))
		}
		prob.should.be.approximately(2, 0.1);
	});

	it('Probabilities of a sentence given sentence should be properly combined from the probabilities of words given sentence ', function() {
		str = random_string

		model.smoothingCoefficient = 1
		model.trainBatch([wordcounts(str)])
	
		str = str.split(" ")
		 
		var prob = Math.exp(model.logProbSentenceGivenSentence(wordcounts(str.join(" ")),wordcounts(str.join(" "))))

		var prob1 = 1
		for (word in str)
		{
			prob1 *= Math.exp(model.logProbWordGivenSentence(str[word], wordcounts(str.join(" "))))
		}
		  
		prob.should.be.approximately(prob1, 0.0001);
	});

	it('Probabilities of a sentence given a dataset should be properly combined from the probabilities of a sentence given a sentence', function() {
		str = random_string

		model.smoothingCoefficient = 1
		model.trainBatch([wordcounts(str)])
	
		str = str.split(" ")
		
		var prob = Math.exp(model.logProbSentenceGivenDataset(wordcounts(str.join(" "))))

		var prob1 = 1
		for (word in str)
		{
			prob1 *= Math.exp(model.logProbWordGivenSentence(str[word], wordcounts(str.join(" "))))
		}
		  
		prob.should.be.approximately(prob1, 0.0001);
	});

	it('produces predictable probabilities', function() {
		model.smoothingCoefficient = 0.9
		model.trainBatch([
			wordcounts("I want aa"),
			wordcounts("I want bb"),
			wordcounts("I want cc")
			]);

		assertProbSentence("I", 0.4444444444444444);//1/3);
		assertProbSentence("I want", 0.19753086419753085);//1/9);
		assertProbSentence("I want aa", 0.04389574759945128);//0.0123456);
		assertProbSentence("I want aa bb", 0.007779301935680535);//0.00026);
		assertProbSentence("I want aa bb cc", 0.0012458805398905997);//0.00000427);
	});

});