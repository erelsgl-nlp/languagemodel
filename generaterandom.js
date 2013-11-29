/**
 * Simple calculation of word-counts in a sentence. 
 * @param sentence
 * @return a hash {word1: count1, word2: count2,...}
 * words are separated by spaces.
 */
module.exports = function(length) {
	var chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghiklmnopqrstuvwxyz';
	length = length ? length : 10;
  var string = '';
  for (var i = 0; i < length; i++) {
    var word_length = Math.floor(Math.random() * 10+1);
    for (var j = 0; j <= word_length; j++)
    {
    	var randomNumber = Math.floor(Math.random() * chars.length);
    	var ch = chars.substring(randomNumber, randomNumber + 1);
    	string += ch
  	}  
  	string += " "
	}
  return string;
}

