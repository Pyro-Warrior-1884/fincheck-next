"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import GraphCard from "./GraphCard"

type ChartItem = {
  model: string
  prediction: number
  confidence: number
  latency_ms: number
  ram_mb: number
  cold_start?: boolean
}

export default function ChartSection({
  data,
}: {
  data: ChartItem[]
}) {
  return (
    <div className="space-y-10">
      {/* GRID OF METRICS */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Confidence */}
        <GraphCard title="Model Confidence (%)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data}>
              <XAxis dataKey="model" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="confidence" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>

        {/* Latency */}
        <GraphCard title="Inference Latency (ms)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data}>
              <XAxis dataKey="model" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="latency_ms" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>

        {/* RAM */}
        <GraphCard title="Memory Usage (MB)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data}>
              <XAxis dataKey="model" hide />
              <YAxis />
              <Tooltip />
              <Bar dataKey="ram_mb" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>

        {/* Predictions */}
        <GraphCard title="Predicted Digit">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data}>
              <XAxis dataKey="model" hide />
              <YAxis allowDecimals={false} domain={[0, 9]} />
              <Tooltip />
              <Bar dataKey="prediction" />
            </BarChart>
          </ResponsiveContainer>
        </GraphCard>
      </div>

      {/* COLD VS WARM START */}
      <GraphCard title="Cold vs Warm Start">
        <div className="flex flex-wrap gap-3">
          {data.map((m) => (
            <span
              key={m.model}
              className={`rounded-full px-4 py-2 text-sm font-medium ${
                m.cold_start
                  ? "bg-red-100 text-red-700"
                  : "bg-green-100 text-green-700"
              }`}
            >
              {m.model} — {m.cold_start ? "Cold Start" : "Warm Start"}
            </span>
          ))}
        </div>
      </GraphCard>
    </div>
  )
}
