//TKEO calculation 

var buffer = [0,0];
var index = 0;

function fillBuffer(input) {
	buffer[index] = (input);
	
	index++;
	index = index % 2;
}

function TKEO(input) {
	var out = 0;
	out = Math.pow(buffer[Math.abs(index - 1)], 2) - (buffer[index] * input);
	fillBuffer(input);
	outlet(0, out*10.);
	//multiply out w 10 to scale up, normally don't do that
}

//debug
function bang() {
	post(buffer);
	post();
}
