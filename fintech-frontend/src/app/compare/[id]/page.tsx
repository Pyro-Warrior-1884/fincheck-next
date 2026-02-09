"use client"

import { useEffect, useState, useMemo } from "react"
import { useParams, useRouter } from "next/navigation"

/* ================= CONSTANTS ================= */

const MODEL_ORDER = [
  "baseline_mnist.pth",
  "kd_mnist.pth",
  "lrf_mnist.pth",
  "pruned_mnist.pth",
  "quantized_mnist.pth",
  "ws_mnist.pth",
]

type ResultDoc = {
  data: {
    MNIST?: Record<string, any>
    CIFAR?: Record<string, any>
  }
}

export default function ComparePage() {
  const { id } = useParams<{ id: string }>()
  const router = useRouter()

  const [doc, setDoc] = useState<ResultDoc | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`/api/results/${id}`)
      .then(r => r.json())
      .then(setDoc)
      .finally(() => setLoading(false))
  }, [id])

  const rows = useMemo(() => {
    if (!doc?.data?.MNIST || !doc?.data?.CIFAR) return []

    return MODEL_ORDER.map(mnistModel => {
      const cifarModel = mnistModel.replace("mnist", "cifar")

      const m = doc.data.MNIST![mnistModel]
      const c = doc.data.CIFAR![cifarModel]

      if (!m || !c) return null

      return {
        model: mnistModel,
        mnist: {
          conf: m.confidence_percent ?? m.confidence_mean ?? 0,
          lat: m.latency_ms ?? m.latency_mean ?? 0,
          ent: m.entropy ?? m.entropy_mean ?? 0,
          risk: m.evaluation?.risk_score ?? 999,
        },
        cifar: {
          conf: c.confidence_percent ?? c.confidence_mean ?? 0,
          lat: c.latency_ms ?? c.latency_mean ?? 0,
          ent: c.entropy ?? c.entropy_mean ?? 0,
          risk: c.evaluation?.risk_score ?? 999,
        },
      }
    }).filter(Boolean)
  }, [doc])

  if (loading) return <p className="p-8">Loading…</p>
  if (!rows.length) return <p className="p-8 text-red-500">No comparison data</p>

  return (
    <div className="mx-auto max-w-7xl p-8 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-semibold">MNIST vs CIFAR Comparison</h1>

        <button
          onClick={() => router.push(`/results/${id}`)}
          className="px-4 py-2 bg-gray-200 rounded"
        >
          ← Back
        </button>
      </div>

      <table className="w-full border text-sm">
        <thead className="bg-gray-100">
          <tr>
            <th className="border p-2">Model</th>
            <th className="border p-2">MNIST Conf</th>
            <th className="border p-2">MNIST Lat</th>
            <th className="border p-2">MNIST Risk</th>
            <th className="border p-2">CIFAR Conf</th>
            <th className="border p-2">CIFAR Lat</th>
            <th className="border p-2">CIFAR Risk</th>
          </tr>
        </thead>

        <tbody>
          {rows.map((r: any) => (
            <tr key={r.model} className="text-center">
              <td className="border p-2 font-medium">{r.model}</td>
              <td className="border p-2">{r.mnist.conf.toFixed(2)}</td>
              <td className="border p-2">{r.mnist.lat.toFixed(3)}</td>
              <td className="border p-2">{r.mnist.risk.toFixed(4)}</td>
              <td className="border p-2">{r.cifar.conf.toFixed(2)}</td>
              <td className="border p-2">{r.cifar.lat.toFixed(3)}</td>
              <td className="border p-2">{r.cifar.risk.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
