"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import ChartWrapper from "../ChartWrapper"

export default function LatencyLive({
  data,
}: {
  data: any[]
}) {
  const series = data.map((d, i) => ({
    step: i + 1,
    latency: d.latency_ms,
    model: d.model,
  }))

  return (
    <ChartWrapper
      title="Latency Time Series"
      description="Inference latency behaviour across execution order"
    >
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={series}>
          <XAxis dataKey="step" />
          <YAxis unit="ms" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="latency"
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
