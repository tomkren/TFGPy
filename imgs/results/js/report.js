function add_row(row) {

    var img_src = row[0];
    var codes = row[1];
    var err = row[2];


    var $tr = $('<tr>').append([
        $('<td>').addClass('txt-cell').text(img_src),
        $('<td>').html($('<img>').attr({src: 'imgs/' + img_src})),
        $('<td>').html($('<img>').attr({src: 'imgs_out/' + img_src})),
        $('<td>').addClass('txt-cell').html(codes[0] + '<br>' + codes[1]),
        $('<td>').addClass('txt-cell').text(err)
    ]);

    $('#report_table').append($tr);
}

function load_rows(num_skip, num_show) {
    var rows_to_show = _.take(_.take(report_data.table, num_skip + num_show), num_show);
    _.each(rows_to_show, function (row) {
        add_row(row);
    });
}

function isScrolledIntoView(el) {
    var rect = el.getBoundingClientRect();
    var elemTop = rect.top;
    var elemBottom = rect.bottom;
    // return (elemTop >= 0) && (elemBottom <= window.innerHeight); // Only completely visible elements return true.
    return elemTop < window.innerHeight && elemBottom >= 0; // Partially visible elements return true.
}

function main() {
    _.each(report_data.stats, function (val, key) {
        var id = '#stats_' + key;
        console.log(id);
        $(id).html($('<b>').text(val));
    });

    var num_show = 200;

    load_rows(0, num_show);
    var num_skip = 0;
    setInterval(function() {
        if (isScrolledIntoView($('#table-tail')[0])) {
            console.log('loading more...');
            num_skip += num_show;
            load_rows(num_skip, num_show);
        }
    }, 200);
}

$(main);