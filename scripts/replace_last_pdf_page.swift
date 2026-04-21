import Foundation
import PDFKit

enum ReplaceError: Error {
    case usage
    case openBase(String)
    case openSlide(String)
    case missingSlidePage
    case writeFailed(String)
}

func run() throws {
    let args = CommandLine.arguments
    guard args.count == 4 else {
        throw ReplaceError.usage
    }

    let baseURL = URL(fileURLWithPath: args[1])
    let slideURL = URL(fileURLWithPath: args[2])
    let outputURL = URL(fileURLWithPath: args[3])

    guard let baseDoc = PDFDocument(url: baseURL) else {
        throw ReplaceError.openBase(baseURL.path)
    }
    guard let slideDoc = PDFDocument(url: slideURL) else {
        throw ReplaceError.openSlide(slideURL.path)
    }
    guard let replacementPage = slideDoc.page(at: 0) else {
        throw ReplaceError.missingSlidePage
    }

    let merged = PDFDocument()
    let keepCount = max(baseDoc.pageCount - 1, 0)

    for idx in 0..<keepCount {
        guard let page = baseDoc.page(at: idx) else { continue }
        merged.insert(page, at: merged.pageCount)
    }
    merged.insert(replacementPage, at: merged.pageCount)

    if FileManager.default.fileExists(atPath: outputURL.path) {
        try FileManager.default.removeItem(at: outputURL)
    }
    if !merged.write(to: outputURL) {
        throw ReplaceError.writeFailed(outputURL.path)
    }
}

do {
    try run()
} catch ReplaceError.usage {
    fputs("Usage: replace_last_pdf_page.swift <base.pdf> <slide.pdf> <output.pdf>\n", stderr)
    exit(2)
} catch ReplaceError.openBase(let path) {
    fputs("Could not open base PDF: \(path)\n", stderr)
    exit(1)
} catch ReplaceError.openSlide(let path) {
    fputs("Could not open slide PDF: \(path)\n", stderr)
    exit(1)
} catch ReplaceError.missingSlidePage {
    fputs("Replacement PDF does not contain page 0\n", stderr)
    exit(1)
} catch ReplaceError.writeFailed(let path) {
    fputs("Failed to write merged PDF: \(path)\n", stderr)
    exit(1)
} catch {
    fputs("Unexpected error: \(error)\n", stderr)
    exit(1)
}
