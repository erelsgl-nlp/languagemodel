var logSumExp = require('./logSumExp');
var LanguageModel = require('./LanguageModel');


/**
 * This class represents a model for two different languages - input language and output language.
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
var CrossLanguageModel = function(opts) {
	this.smoothingCoefficient = opts.smoothingCoefficient || 0.9;
	this.inputLanguageModel = new LanguageModel(opts);
	this.outputLanguageModel = new LanguageModel(opts);
}

CrossLanguageModel.prototype = {

	/**
	 * Tell the classifier that the given sample belongs to the given classes.
	 * 
	 * @param sample
	 *            a document.
	 * @param classes
	 *            an object whose KEYS are classes, or an array whose VALUES are classes.
	 */
	trainOnline: function(sample, classes) {
		throw new Error("CrossLanguageModel does not support online training");
	},

	/**
	 * Train the classifier with all the given documents.
	 * 
	 * @param dataset
	 *            an array with objects of the format: 
	 *            {input: {feature1:count1, feature2:count2,...}, output: {feature1:count1, feature2:count2,...}}
	 */
	trainBatch : function(dataset) {
		this.inputLanguageModel.trainBatch(dataset.map(function(datum) {return datum.input;}));
		this.outputLanguageModel.trainBatch(dataset.map(function(datum) {return datum.output;}));
		this.dataset = dataset;
	},
	
	getAllWordCounts: function() {
		return this.mapWordToTotalCount;
	},

	/**
	 * Use the model trained so far to classify a new sample.
	 * 
	 * @param segment a part of a text sentence.
	 * @param explain - int - if positive, an "explanation" field, with the given length, will be added to the result.
	 *  
	 * @return an array whose VALUES are classes.
	 */
	classify: function(sentence, explain) {
		
	},
	
	/**
	 * Calculate the similarity scores (minus-divergences) between the given input sentence and all output sentences in the corpus, sorted from high (most similar) to low (least similar). 
	 */
	similarities: function(inputSentenceCounts) {
		var sims = [];
		for (var i in this.dataset) {
			var output = this.dataset[i].output;
			sims.push({output: output, similarity:  -this.divergence(inputSentenceCounts, output)});
		}
		sims.sort(function(a,b) {
			return b.similarity-a.similarity;
		});
		return sims;
	},
	
	/**
	 * Calculate the Kullback-Leibler divergence between the language models of the given samples.
	 * This can be used as an approximation of the (inverse) semantic similarity. between them. 
	 *
	 * @param inputSentenceCounts hash that represents a sentence from the INPUT domain.
	 * @param outputSentenceCounts hash that represents a sentence from the OUTPUT domain.
	 * 
	 * @note divergence is not symmetric - divergence(a,b) != divergence(b,a).
	 */
	divergence: function(inputSentenceCounts, outputSentenceCounts) {         // (6)   D(P(W)||P(F)) = ...
		var elements = [];
		for (var feature in outputSentenceCounts) {
			if (feature=='_total') continue;
			var logFeatureGivenInput  = this.logProbFeatureGivenSentence(feature, inputSentenceCounts);
			if (isNaN(logFeatureGivenInput)||!isFinite(logFeatureGivenInput)) throw new Error("logFeatureGivenInput is "+logFeatureGivenInput);
			var logFeatureGivenOutput = this.outputLanguageModel.logProbWordGivenSentence(feature, outputSentenceCounts);
			if (isNaN(logFeatureGivenOutput)||!isFinite(logFeatureGivenOutput)) throw new Error("logFeatureGivenOutput ("+feature+", "+outputSentenceCounts+") is "+logFeatureGivenOutput);
			var probFeatureGivenInput = Math.exp(logFeatureGivenInput);
			var element = probFeatureGivenInput * (logFeatureGivenInput - logFeatureGivenOutput);
			if (isNaN(element)||!isFinite(element)) throw new Error(probFeatureGivenInput+" * ("+logFeatureGivenInput+" - "+logFeatureGivenOutput+") = "+element);
			console.log(probFeatureGivenInput+" * ("+logFeatureGivenInput+" - "+logFeatureGivenOutput+") = "+element);
			elements.push(element)
		}
		//console.dir(elements);
		return elements.reduce(function(memo, num){ return memo + num; }, 0);
	},
	
	/**
	 * @param feature a single feature (-word) from the OUTPUT domain.
	 * @param givenSentenceCounts a hash that represents a sentence from the INPUT domain.
	 */
	logProbFeatureGivenSentence: function(feature, givenSentenceCounts) {  // (5) P(f|W) = ...
		var logSentenceAndFeature = this.logProbSentenceAndFeatureGivenDataset(feature,givenSentenceCounts);
		if (isNaN(logSentenceAndFeature)||!isFinite(logSentenceAndFeature)) throw new Error("logSentenceAndFeature is "+logSentenceAndFeature);
		var logSentence = this.inputLanguageModel.logProbSentenceGivenDataset(givenSentenceCounts);
		if (isNaN(logSentence)||!isFinite(logSentence)) throw new Error("logSentence is "+logSentence);
		return logSentenceAndFeature - logSentence;
	},
	
	/**
	 * @param feature a single feature (-word) from the OUTPUT domain.
	 * @param sentenceCounts a hash that represents a sentence from the INPUT domain.
	 */
	logProbSentenceAndFeatureGivenDataset: function(feature, sentenceCounts) {  // (2') log P(f,w1...wn) = ...
		var logProducts = [];
		for (var i in this.dataset) {
			var datum = this.dataset[i];
			logProducts.push(
				this.inputLanguageModel .logProbSentenceGivenSentence(sentenceCounts, datum.input) +
				this.outputLanguageModel.logProbWordGivenSentence(feature, datum.output)
				);
		}
		//console.dir(logProducts);
		var logSentenceLikelihood = logSumExp(logProducts);
		
		return logSentenceLikelihood - Math.log(this.inputLanguageModel.dataset.length);
	},
}




/*
 * UTILITY FUNCTIONS
 */


/**
 * @return log(exp(a)+exp(b)) 
 * @note handles large numbers robustly.        
 */
function logSumExp(a, b) {
	if (a>b) {
		if (b-a>-10)
			return a + Math.log(1+Math.exp(b-a));
		else
			return a;
	} else {
		if (a-b>-10)
			return b + Math.log(1+Math.exp(a-b));
		else
			return b;
	}
}


/**
 * @param a vector of numbers.
 * @return log(sum[i=1..n](exp(ai))) = 
 *         m + log(sum[i=1..n](exp(ai-m)))
 * Where m = max[i=1..n](ai)
 * @note handles large numbers robustly.        
 */
function logSumExp(a) {
	var m = Math.max.apply(null,a);
	var sum = 0;
	for (var i=0; i<a.length; ++i)
		if (a[i]>m-10)
			sum += Math.exp(a[i]-m);
	return m + Math.log(sum);
}

module.exports = CrossLanguageModel;

if (process.argv[1] === __filename) {
	console.log("CrossLanguageModel demo start");
	
	var classifier = new CrossLanguageModel({
		smoothingFactor : 0.9,
	});
	
	function wordcounts(sentence) {
		return sentence.split(' ').reduce(function(counts, word) {
		    counts[word] = (counts[word] || 0) + 1;
		    return counts;
		  }, {});
	}
	
	classifier.trainBatch([
		{input: wordcounts("I want aa"), output: wordcounts("a")},
		{input: wordcounts("I want bb"), output: wordcounts("b")},
		{input: wordcounts("I want cc"), output: wordcounts("c")},
	    ]);
	
	var test = function(sentence) {
		console.log(sentence+": ");
		console.log(classifier.similarities(wordcounts(sentence)));
	}
	
	//test("I want");
	test("I want aa");
	//test("I want bb");
	test("I want aa bb");
	//test("I want aa bb cc");
	
	
	console.log("CrossLanguageModel demo end");
}
