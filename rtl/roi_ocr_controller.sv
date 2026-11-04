// OCR back-end for one buffered plate ROI.  The input is a binary, rectified
// ROI stream.  It writes the ROI to local memory while accumulating its column
// projection, then segments, resamples and classifies up to MAX_CHARS glyphs.
// A production design can replace roi_mem with a true dual-port BRAM or DMA.
module roi_ocr_controller #(
    parameter int MAX_ROI_WIDTH = 192,
    parameter int MAX_ROI_HEIGHT = 64,
    parameter int MAX_CHARS = 8,
    parameter int CLASS_COUNT = 67,
    parameter string TEMPLATE_MEM_FILE = ""
) (
    input  logic clk, rst_n,
    input  logic roi_in_valid, roi_in_sof, roi_in_eol, roi_in_eof, roi_in_pixel,
    input  logic [$clog2(MAX_ROI_HEIGHT+1)-1:0] foreground_threshold,
    input  logic [$clog2(MAX_ROI_WIDTH):0] min_char_width,
    output logic result_valid,
    output logic [$clog2(MAX_CHARS)-1:0] result_index,
    output logic [$clog2(CLASS_COUNT)-1:0] result_class,
    output logic plate_done,
    output logic [$clog2(MAX_CHARS+1)-1:0] plate_length,
    output logic busy
);
    localparam XW = $clog2(MAX_ROI_WIDTH);
    localparam YW = $clog2(MAX_ROI_HEIGHT);
    localparam CW = $clog2(MAX_CHARS);
    typedef enum logic [2:0] {WAIT_PROJECTION, SEGMENT, START_RESAMPLE,
                              WAIT_RESAMPLE, START_CLASSIFY, WAIT_CLASSIFY} state_t;
    state_t state;

    logic roi_mem [0:MAX_ROI_WIDTH*MAX_ROI_HEIGHT-1];
    logic [XW-1:0] write_x;
    logic [YW-1:0] write_y;
    logic projection_valid, projection_eof_seen, projection_consumed;
    logic [XW-1:0] projection_addr;
    logic [YW:0] projection_count;
    logic [XW:0] roi_width;
    logic [YW:0] roi_height;
    logic segment_start, segment_char_valid, segment_done;
    logic [XW-1:0] segment_x_start, segment_x_end;
    logic [3:0] segment_char_index;
    logic [XW-1:0] char_starts [0:MAX_CHARS-1];
    logic [XW-1:0] char_ends [0:MAX_CHARS-1];
    logic [CW:0] char_count;
    logic [CW:0] final_char_count;
    logic [CW-1:0] process_index;
    logic resample_start, resample_done, resample_busy;
    logic [XW-1:0] source_x;
    logic [YW-1:0] source_y;
    logic source_pixel;
    logic [511:0] sample_bits;
    logic classify_start, classifier_busy, classifier_done;
    logic [$clog2(CLASS_COUNT)-1:0] classifier_class;
    logic [9:0] classifier_distance;

    always_comb begin
        final_char_count = char_count;
        if (segment_char_valid && segment_char_index < MAX_CHARS)
            final_char_count = segment_char_index + 1'b1;
    end

    // ROI write port.  The read port below is independent and is inferred as
    // simple dual-port memory by Vivado when roi_mem is mapped to BRAM.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            write_x <= '0;
            write_y <= '0;
            projection_eof_seen <= 1'b0;
        end else if (roi_in_valid) begin
            if (roi_in_sof) begin
                write_x <= '0;
                write_y <= '0;
                projection_eof_seen <= 1'b0;
            end
            roi_mem[write_y*MAX_ROI_WIDTH + write_x] <= roi_in_pixel;
            if (roi_in_eol) begin
                write_x <= '0;
                if (roi_in_eof) begin
                    write_y <= '0;
                    projection_eof_seen <= 1'b1;
                end else begin
                    write_y <= write_y + 1'b1;
                end
            end else begin
                write_x <= write_x + 1'b1;
            end
        end
    end
    always_comb source_pixel = roi_mem[source_y*MAX_ROI_WIDTH + source_x];

    char_projection #(.MAX_ROI_WIDTH(MAX_ROI_WIDTH), .MAX_ROI_HEIGHT(MAX_ROI_HEIGHT)) u_projection (
        .clk(clk), .rst_n(rst_n), .in_valid(roi_in_valid), .in_sof(roi_in_sof),
        .in_eol(roi_in_eol), .in_eof(roi_in_eof), .in_pixel(roi_in_pixel),
        .projection_valid(projection_valid), .projection_rd_addr(projection_addr),
        .projection_rd_count(projection_count), .roi_width(roi_width), .roi_height(roi_height)
    );
    char_segmenter #(.MAX_ROI_WIDTH(MAX_ROI_WIDTH), .MAX_ROI_HEIGHT(MAX_ROI_HEIGHT)) u_segmenter (
        .clk(clk), .rst_n(rst_n), .start(segment_start), .roi_width(roi_width),
        .projection_addr(projection_addr), .projection_count(projection_count),
        .foreground_threshold(foreground_threshold), .min_char_width(min_char_width),
        .char_valid(segment_char_valid), .char_x_start(segment_x_start),
        .char_x_end(segment_x_end), .char_index(segment_char_index), .done(segment_done)
    );
    char_resampler #(.MAX_ROI_WIDTH(MAX_ROI_WIDTH), .MAX_ROI_HEIGHT(MAX_ROI_HEIGHT)) u_resampler (
        .clk(clk), .rst_n(rst_n), .start(resample_start),
        .char_x_start(char_starts[process_index]), .char_x_end(char_ends[process_index]),
        .roi_height(roi_height), .source_x(source_x), .source_y(source_y),
        .source_pixel(source_pixel), .busy(resample_busy), .done(resample_done), .sample_bits(sample_bits)
    );
    plate_template_classifier #(.CLASS_COUNT(CLASS_COUNT), .MEM_INIT_FILE(TEMPLATE_MEM_FILE)) u_classifier (
        .clk(clk), .rst_n(rst_n), .start(classify_start), .sample_bits(sample_bits),
        .busy(classifier_busy), .done(classifier_done), .class_id(classifier_class),
        .hamming_distance(classifier_distance)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= WAIT_PROJECTION;
            segment_start <= 1'b0;
            resample_start <= 1'b0;
            classify_start <= 1'b0;
            result_valid <= 1'b0;
            result_index <= '0;
            result_class <= '0;
            plate_done <= 1'b0;
            plate_length <= '0;
            char_count <= '0;
            process_index <= '0;
            busy <= 1'b0;
            projection_consumed <= 1'b0;
        end else begin
            segment_start <= 1'b0;
            resample_start <= 1'b0;
            classify_start <= 1'b0;
            result_valid <= 1'b0;
            plate_done <= 1'b0;
            if (roi_in_sof && roi_in_valid) begin
                char_count <= '0;
                plate_length <= '0;
                busy <= 1'b1;
                projection_consumed <= 1'b0;
            end
            case (state)
                WAIT_PROJECTION: begin
                    if (projection_valid && projection_eof_seen && !projection_consumed) begin
                        segment_start <= 1'b1;
                        projection_consumed <= 1'b1;
                        state <= SEGMENT;
                    end
                end
                SEGMENT: begin
                    if (segment_char_valid && segment_char_index < MAX_CHARS) begin
                        char_starts[segment_char_index[CW-1:0]] <= segment_x_start;
                        char_ends[segment_char_index[CW-1:0]] <= segment_x_end;
                        char_count <= segment_char_index + 1'b1;
                    end
                    if (segment_done) begin
                        process_index <= '0;
                        plate_length <= final_char_count;
                        if (final_char_count == 0) begin
                            plate_done <= 1'b1;
                            busy <= 1'b0;
                            state <= WAIT_PROJECTION;
                        end else begin
                            state <= START_RESAMPLE;
                        end
                    end
                end
                START_RESAMPLE: begin
                    resample_start <= 1'b1;
                    state <= WAIT_RESAMPLE;
                end
                WAIT_RESAMPLE: begin
                    if (resample_done) begin
                        state <= START_CLASSIFY;
                    end
                end
                START_CLASSIFY: begin
                    classify_start <= 1'b1;
                    state <= WAIT_CLASSIFY;
                end
                WAIT_CLASSIFY: begin
                    if (classifier_done) begin
                        result_valid <= 1'b1;
                        result_index <= process_index;
                        result_class <= classifier_class;
                        if (process_index + 1'b1 == char_count) begin
                            plate_length <= char_count;
                            plate_done <= 1'b1;
                            busy <= 1'b0;
                            state <= WAIT_PROJECTION;
                        end else begin
                            process_index <= process_index + 1'b1;
                            state <= START_RESAMPLE;
                        end
                    end
                end
                default: state <= WAIT_PROJECTION;
            endcase
        end
    end
endmodule
