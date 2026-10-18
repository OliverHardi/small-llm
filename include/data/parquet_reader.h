#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include "tokenizer.h"

std::string readParquet(const std::string& path) {
    std::cout << "Reading file " + path << std::endl;
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    auto reader_properties = parquet::ReaderProperties(pool);
    reader_properties.set_buffer_size(4096 * 4);
    reader_properties.enable_buffered_stream();

    auto arrow_reader_props = parquet::ArrowReaderProperties();
    arrow_reader_props.set_batch_size(10000);

    parquet::arrow::FileReaderBuilder reader_builder;

    auto open_status = reader_builder.OpenFile(path, /*memory_map=*/false, reader_properties);

    reader_builder.memory_pool(pool);
    reader_builder.properties(arrow_reader_props);

    auto build_result = reader_builder.Build();

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader = std::move(*build_result);

    auto rb_reader_result = arrow_reader->GetRecordBatchReader();

    std::unique_ptr<arrow::RecordBatchReader> rb_reader = std::move(*rb_reader_result);

    std::string str = "";
    str.reserve(1000000);

    for (arrow::Result<std::shared_ptr<arrow::RecordBatch>> maybe_batch : *rb_reader) {
        auto batch = *maybe_batch;

        for (int64_t i = 0; i < batch->num_rows(); ++i) {
            auto scalar_result = batch->column(0)->GetScalar(i);
            str.append( (*scalar_result)->ToString() + '\n' );

            // std::cout << text << std::endl;
        }
    }
    str.shrink_to_fit();

    std::cout << "Done reading file " + path << std::endl;

    return str;
}

void tokenizeParquet(const std::string& path, Tokenizer& tokenizer) {
    namespace fs = std::filesystem;
    std::cout << "Tokenizing file: " << path << std::endl;

    fs::path input_path(path);
    fs::path output_path = input_path;
    output_path.replace_extension(".bin");

    std::ofstream output(
        output_path,
        std::ios::binary | std::ios::out
    );

    if (!output.is_open()) {
        throw std::runtime_error(
            "Could not open output file: " +
            output_path.string()
        );
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();

    auto reader_properties = parquet::ReaderProperties(pool);
    reader_properties.set_buffer_size(4096 * 4);
    reader_properties.enable_buffered_stream();

    auto arrow_reader_props = parquet::ArrowReaderProperties();
    arrow_reader_props.set_batch_size(10000);

    parquet::arrow::FileReaderBuilder reader_builder;

    auto open_status = reader_builder.OpenFile(path, /*memory_map=*/false, reader_properties);

    reader_builder.memory_pool(pool);
    reader_builder.properties(arrow_reader_props);

    auto build_result = reader_builder.Build();

    std::unique_ptr<parquet::arrow::FileReader> arrow_reader = std::move(*build_result);

    auto rb_reader_result = arrow_reader->GetRecordBatchReader();

    std::unique_ptr<arrow::RecordBatchReader> rb_reader = std::move(*rb_reader_result);

    uint32_t eos_token = tokenizer.getEosToken();

    uint64_t total_tokens = 0;
    uint64_t total_documents = 0;

    // ------------------------------------------------------------
    // Stream through RecordBatches
    // ------------------------------------------------------------

    for (auto maybe_batch : *rb_reader) {

        if (!maybe_batch.ok()) {
            throw std::runtime_error(
                "Error reading RecordBatch: " +
                maybe_batch.status().ToString()
            );
        }

        auto batch = maybe_batch.ValueOrDie();

        // Assuming column 0 contains the text.
        auto array =
            std::static_pointer_cast<arrow::StringArray>(
                batch->column(0)
            );

        // --------------------------------------------------------
        // Process every text row in this batch
        // --------------------------------------------------------

        for (int64_t i = 0; i < array->length(); ++i) {

            // Skip null text values
            if (array->IsNull(i)) {
                continue;
            }

            // Get the text without creating an Arrow Scalar.
            std::string_view text = array->GetView(i);

            // Tokenize
            std::vector<uint32_t> tokens =
                tokenizer.encode(std::string(text));

            // ----------------------------------------------------
            // Write token IDs
            // ----------------------------------------------------

            if (!tokens.empty()) {
                output.write(
                    reinterpret_cast<const char*>(tokens.data()),
                    tokens.size() * sizeof(uint32_t)
                );

                if (!output) {
                    throw std::runtime_error(
                        "Error writing token data."
                    );
                }

                total_tokens += tokens.size();
            }

            // ----------------------------------------------------
            // Write EOS after every document/row
            // ----------------------------------------------------

            output.write(
                reinterpret_cast<const char*>(&eos_token),
                sizeof(eos_token)
            );

            if (!output) {
                throw std::runtime_error(
                    "Error writing EOS token."
                );
            }

            ++total_tokens;
            ++total_documents;
        }

        std::cout
            << "\rDocuments: "
            << total_documents
            << " | Tokens: "
            << total_tokens
            << std::flush;
    }

    output.close();

    std::cout << std::endl;

    std::cout
        << "Done tokenizing "
        << path
        << std::endl;

    std::cout
        << "Documents: "
        << total_documents
        << std::endl;

    std::cout
        << "Tokens: "
        << total_tokens
        << std::endl;

    std::cout
        << "Output: "
        << output_path
        << std::endl;

}