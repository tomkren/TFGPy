

function isScrolledIntoView(el) {
    var rect = el.getBoundingClientRect();
    var elemTop = rect.top;
    var elemBottom = rect.bottom;
    // return (elemTop >= 0) && (elemBottom <= window.innerHeight); // Only completely visible elements return true.
    return elemTop < window.innerHeight && elemBottom >= 0; // Partially visible elements return true.
}

function isFloat(n){
    return Number(n) === n && n % 1 !== 0;
}

function MkRowsLoader($table_container, rows) {

    var num_skip = 0;
    var num_show = 200;
    var interval = 200;
    var $table;

    function init() {
        create_empty_table();
        load_rows(0, num_show);
        setInterval(function() {
            if (isScrolledIntoView($('#table-tail')[0])) {
                num_skip += num_show;
                console.log('loading more... load_rows('+num_skip+', '+num_show+')');
                load_rows(num_skip, num_show);
            }
        }, interval);
    }

    function shuffle() {
        rows = _.shuffle(rows);
        num_skip = 0;

        create_empty_table();
        load_rows(0, num_show);
    }

    function create_empty_table() {

        $table = $('<table>').html(
                $('<tr>').append([
                    $('<th>').text(''),
                    $('<th>').text('file'),
                    $('<th>').text('in'),
                    $('<th>').text('out'),
                    $('<th>').text('raw output / original input'),
                    $('<th>').text('error')
        ]));

        $table_container.html($table);
    }

    function add_row(i, row) {

        var img_src = row[0];
        var codes = row[1];
        var err = row[2];


        var $tr = $('<tr>').append([
            $('<td>').addClass('txt-cell').text(i),
            $('<td>').addClass('txt-cell').text(img_src),
            $('<td>').html($('<img>').attr({src: 'imgs/' + img_src})),
            $('<td>').html($('<img>').attr({src: 'imgs_out/' + img_src})),
            $('<td>').addClass('txt-cell').html('output &nbsp; : ' + codes[0] + '<br>' + 'original : ' +codes[1]),
            $('<td>').addClass('txt-cell').text(err)
        ]);

        $table.append($tr);
    }

    function load_rows(num_skip, num_show) {
        var rows_to_show = _.drop(_.take(rows, num_skip + num_show), num_skip);
        var i = num_skip+1;
        _.each(rows_to_show, function (row) {
            add_row(i, row);
            i++;
        });
    }



    init();
    return {shuffle: shuffle};
}

function main() {

    var precision = 3;

    _.each(report_data.stats, function (val, key) {
        var id = '#stats_' + key;
        console.log(id);

        $(id).html($('<b>').text(isFloat(val) ? val.toFixed(precision) : val));
    });

    var rowsLoader = MkRowsLoader($('#report_table_container'), report_data.table);

    $('#shuffle-button').click(function () {
        console.log('shuffling!');
        rowsLoader.shuffle();
    });

}

$(main);