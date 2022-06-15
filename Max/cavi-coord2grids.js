autowatch = 1;
inlets = 1;
outlets = 1;

// function msg_int(val) {
//     switch (inlet) {
//         default:
//             post("left inlet: " + val + "\n");
//     }
// }


function to_grid(x, y, edt) {
    var grid_nr = 0
    //function that map the touch interface coordinates 
    //to the respective grids in the sequencer
    for (var i = 0; i <= 16; i++) {
        grid_nr = x + (y * i)
    }
    outlet(0, grid_nr)
    // send bang to the respective grids
    send = new Global("xyz");
    r_name = ("g" + grid_nr.toString());
    outlet(0, r_name);
    // post(r_name + "\n")
    if (edt == 0) {
        send.msg = 0;
    } else {
        send.msg = 1;
    }
    send.sendnamed(r_name, "msg");
}



// function to_grid(x, y, edt) {
//     var grid_nr = 0;
//     if (y == 0) {
//         grid_nr = x;
//     } else if (y == 1) {
//         grid_nr = x + (16 * y);
//     } else if (y == 2) {
//         grid_nr = x + (16 * y);
//     } else if (y == 3) {
//         grid_nr = x + (16 * y);
//     } else if (y == 4) {
//         grid_nr = x + (16 * y);
//     } else if (y == 5) {
//         grid_nr = x + (16 * y);
//     } else if (y == 6) {
//         grid_nr = x + (16 * y);
//     } else if (y == 7) {
//         grid_nr = x + (16 * y);
//     } else if (y == 8) {
//         grid_nr = x + (16 * y);
//     } else if (y == 9) {
//         grid_nr = x + (16 * y);
//     } else if (y == 10) {
//         grid_nr = x + (16 * y);
//     } else if (y == 11) {
//         grid_nr = x + (16 * y);
//     } else if (y == 12) {
//         grid_nr = x + (16 * y);
//     } else if (y == 13) {
//         grid_nr = x + (16 * y);
//     } else if (y == 14) {
//         grid_nr = x + (16 * y);
//     } else if (y == 15) {
//         grid_nr = x + (16 * y);
//     }


//     // send bang to the respective grids
//     send = new Global("xyz");
//     r_name = ("g" + grid_nr.toString());
//     outlet(0, r_name);
//     // post(r_name + "\n")
//     if (edt == 0) {
//         send.msg = 0;
//     } else {
//         send.msg = 1;
//     }
//     send.sendnamed(r_name, "msg");
// }




function reset_seq() {
    s = new Global("xyz");
    s.off = 0;
    for (var i = 0; i < 256; i++) {
        s.sendnamed("g" + i.toString(), "off")
    }
}