// Sequential binary-template classifier for normalized 16x32 character masks.
// A 67-class Chinese licence-plate alphabet requires only 34,304 template bits
// (about 4.2 KiB).  The comparison is serialized over 32-bit words so it uses
// a small, deterministic amount of logic rather than 67 full popcount trees.
module plate_template_classifier #(
    parameter int CHAR_WIDTH = 16,
    parameter int CHAR_HEIGHT = 32,
    parameter int CLASS_COUNT = 67,
    parameter string MEM_INIT_FILE = ""
) (
    input  logic clk, rst_n,
    input  logic start,
    input  logic [CHAR_WIDTH*CHAR_HEIGHT-1:0] sample_bits,
    output logic busy,
    output logic done,
    output logic [$clog2(CLASS_COUNT)-1:0] class_id,
    output logic [$clog2(CHAR_WIDTH*CHAR_HEIGHT+1)-1:0] hamming_distance
);
    localparam int INPUT_BITS = CHAR_WIDTH * CHAR_HEIGHT;
    localparam int WORD_BITS = 32;
    localparam int WORDS = (INPUT_BITS + WORD_BITS - 1) / WORD_BITS;
    localparam int CLASS_W = $clog2(CLASS_COUNT);
    localparam int WORD_W = (WORDS <= 1) ? 1 : $clog2(WORDS);
    localparam int SCORE_W = $clog2(INPUT_BITS + 1);

    logic [INPUT_BITS-1:0] templates [0:CLASS_COUNT-1];
    logic [INPUT_BITS-1:0] sample_reg;
    logic [CLASS_W-1:0] scan_class, best_class;
    logic [WORD_W-1:0] scan_word;
    logic [SCORE_W-1:0] accumulated, best_score;
    logic [31:0] xor_word;
    logic [SCORE_W-1:0] word_distance, score_with_word;
    integer i;

    // Vivado accepts a $readmemh-initialized BRAM. Each line of the .mem file
    // is one LSB-first packed normalized glyph, padded to INPUT_BITS if needed.
    initial begin
        if (MEM_INIT_FILE != "")
            $readmemh(MEM_INIT_FILE, templates);
    end

    function automatic [5:0] popcount32(input logic [31:0] value);
        integer k;
        begin
            popcount32 = '0;
            for (k = 0; k < 32; k = k + 1)
                popcount32 = popcount32 + value[k];
        end
    endfunction

    always_comb begin
        xor_word = sample_reg[scan_word*WORD_BITS +: WORD_BITS] ^
                   templates[scan_class][scan_word*WORD_BITS +: WORD_BITS];
        word_distance = popcount32(xor_word);
        score_with_word = accumulated + word_distance;
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            class_id <= '0;
            hamming_distance <= '0;
            scan_class <= '0;
            scan_word <= '0;
            sample_reg <= '0;
            accumulated <= '0;
            best_class <= '0;
            best_score <= {SCORE_W{1'b1}};
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                sample_reg <= sample_bits;
                busy <= 1'b1;
                scan_class <= '0;
                scan_word <= '0;
                accumulated <= '0;
                best_class <= '0;
                best_score <= {SCORE_W{1'b1}};
            end else if (busy) begin
                if (scan_word == WORDS-1) begin
                    if (score_with_word < best_score) begin
                        best_score <= score_with_word;
                        best_class <= scan_class;
                    end
                    accumulated <= '0;
                    scan_word <= '0;
                    if (scan_class == CLASS_COUNT-1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        if (score_with_word < best_score) begin
                            class_id <= scan_class;
                            hamming_distance <= score_with_word;
                        end else begin
                            class_id <= best_class;
                            hamming_distance <= best_score;
                        end
                    end else begin
                        scan_class <= scan_class + 1'b1;
                    end
                end else begin
                    accumulated <= score_with_word;
                    scan_word <= scan_word + 1'b1;
                end
            end
        end
    end
endmodule
