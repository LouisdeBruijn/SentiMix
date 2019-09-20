const request = require("request");
const fs = require('fs');
const dir = "../data/"

//Don't use these too extensively only 250 requests per month
const tokens = {
	consumer_key: "eWv49LeLWhuYMfzf7KSXNS1gV",
	consumer_secret: "aBSWx0X6dKixKJ34azM1ZQ4M8Nz6krEUEi1WXBFgclR06SlIxf"
}

var searchUrl = "https://api.twitter.com/1.1/tweets/search/30day/dev.json"

function search(url, query, fileCount) {
	var body = {
		"query": query
	}

	var headers = {
		"content-type": "application/json"
	}

	request.post(url, {
		oauth: tokens,
		json: true,
		headers: headers,
		body: body
	}, (err, res, body) => {

		if (err) {
			console.log(err)
		} else {
			console.log("success")
			var jsn = JSON.stringify(body);
			var filename = "output" + fileCount + ".json";
			write(filename, jsn);
		}
	})
}


function write(filename, data) {
	var path = dir + filename + ".json";
	fs.writeFile(path, data, (err) => {
		if (err)
			console.log(err);
	})
}

function start() {

	var path = dir + "log.json"

	var log = {
		fileCount: 0
	}

	fs.exists(path, (exists) => {
		if (exists) {
			fs.readFile(path, (err, data) => {
				if (err) {
					console.log(err)
				} else {
					log = JSON.parse(data)
				}
			})
		} else {
			write("log", JSON.stringify(log))
		}
	})

	input("Be careful for what you wish for (only 250 requests per month):", function(data){
		search(searchUrl, data, log.fileCount)
		log.fileCount ++;
		write("log", JSON.stringify(log))
	})
}

function input(prompt, callback) {
	// Get process.stdin as the standard input object.
	var standard_input = process.stdin;

	// Set input character encoding.
	standard_input.setEncoding('utf-8');

	// Prompt user to input data in console.
	console.log(prompt);

	// When user input data and click enter key.
	standard_input.on('data', function (data) {
		// User input exit.
		if (data === 'exit\n') {
			// Program exit.
			console.log("User input complete, program exit.");
			process.exit();
		} else {
			// Print user input in console.
			callback(data)
		}
	});
}

start()