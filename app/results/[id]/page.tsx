"use client"

import { useEffect, useState } from "react"
import { useParams } from "next/navigation"
import ChartSection from "@/components/ChartSection"

type ResultDoc = {
  data: Record<
    string,
    {
      confidence?: number
    }
  >
}

export default function ResultPage() {
  const params = useParams()
  const id = params?.id as string

  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return

    fetch(`/api/results/${id}`)
      .then(async (r) => {
        if (!r.ok) throw new Error("Failed")
        return r.json()
      })
      .then((data) => {
        setDoc(data)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [id])

  if (loading) {
    return <p className="p-8">Loading results…</p>
  }

  if (!doc || !doc.data) {
    return (
      <p className="p-8 text-red-600">
        Failed to load results
      </p>
    )
  }

  /**
   * 🔹 Transform backend data → chart data
   * Safely filters invalid values
   */
  const chartData = Object.entries(doc.data)
    .map(([model, value]) => ({
      model,
      confidence:
        typeof value?.confidence === "number"
          ? value.confidence
          : 0,
    }))
    .filter((item) => item.confidence > 0)

  return (
    <div className="mx-auto max-w-5xl p-8 space-y-10">
      <h1 className="text-3xl font-bold tracking-tight">
        Inference Results
      </h1>

      {/* 📊 Chart */}
      {chartData.length > 0 && (
        <ChartSection data={chartData} />
      )}

      {/* 📄 Raw output */}
      <div className="rounded-xl border bg-gray-50 p-6">
        <h2 className="mb-3 text-lg font-semibold">
          Raw Model Output
        </h2>
        <pre className="overflow-auto text-sm text-gray-800">
          {JSON.stringify(doc.data, null, 2)}
        </pre>
      </div>
    </div>
  )
}
