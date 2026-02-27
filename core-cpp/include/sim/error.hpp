#pragma once

#include <stdexcept>
#include <string>

enum class ExitCode : int {
    SUCCESS = 0,
    E_ARG_MISSING = 2,
    E_ARG_INVALID = 3,
    E_CASE_PARSE = 4,
    E_CASE_SCHEMA = 5,
    E_IO = 6,
};

class CliError : public std::runtime_error {
public:
    CliError(ExitCode code, std::string symbol, const std::string& message)
        : std::runtime_error(message), code_(code), symbol_(std::move(symbol)) {}

    ExitCode code() const { return code_; }
    const std::string& symbol() const { return symbol_; }

private:
    ExitCode code_;
    std::string symbol_;
};

