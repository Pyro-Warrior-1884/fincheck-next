import Link from "next/link"
import connectDB from "@/lib/mongodb"

export const dynamic = "force-dynamic"

type ResultDoc = {
  _id: any
  createdAt: Date
  data: Record<
    string,
    {
      confidence?: number
    }
  >
}

export default async function ResultsPage() {
  const db = await connectDB()
  const docs = (await db
    .collection("model_results")
    .find()
    .sort({ createdAt: -1 })
    .toArray()) as ResultDoc[]

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-8">
      <h1 className="text-3xl font-bold tracking-tight">
        Inference Results
      </h1>

      <div className="space-y-4">
        {docs.map((doc) => {
          const models = Object.entries(doc.data || [])
            .map(([model, v]) => ({
              model,
              confidence:
                typeof v?.confidence === "number"
                  ? v.confidence
                  : 0,
            }))
            .sort((a, b) => b.confidence - a.confidence)

          const best = models[0]

          return (
            <Link
              key={doc._id.toString()}
              href={`/results/${doc._id}`}
              className="group block rounded-xl border border-gray-200 bg-white p-5
                         transition-all hover:shadow-md hover:border-gray-300
                         dark:border-gray-800 dark:bg-gray-900"
            >
              {/* Header */}
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date(doc.createdAt).toLocaleString()}
                </p>

                {best && (
                  <span className="rounded-full bg-green-100 px-3 py-1
                                   text-xs font-semibold text-green-700
                                   dark:bg-green-900/30 dark:text-green-400">
                    🥇 Best: {best.model}
                  </span>
                )}
              </div>

              {/* Ranking */}
              <div className="mt-4 flex flex-wrap gap-2">
                {models.map((m, idx) => (
                  <span
                    key={m.model}
                    className={`rounded-md px-3 py-1 text-xs font-medium
                      ${
                        idx === 0
                          ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400"
                          : idx === 1
                          ? "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
                          : "bg-gray-50 text-gray-500 dark:bg-gray-700 dark:text-gray-400"
                      }`}
                  >
                    #{idx + 1} {m.model} ·{" "}
                    {Math.round(m.confidence * 100)}%
                  </span>
                ))}
              </div>
            </Link>
          )
        })}
      </div>
    </div>
  )
}
