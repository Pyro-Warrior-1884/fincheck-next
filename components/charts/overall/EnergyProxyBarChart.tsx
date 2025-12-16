"use client"

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import ChartWrapper from "../ChartWrapper"

type Item = {
  model: string
  cpu_usage?: number
  latency_ms: number
}

export default function EnergyProxyBarChart({
  data,
}: {
  data: Item[]
}) {
  const energyData = data.map((d) => ({
    model: d.model,
    energy_proxy:
      d.cpu_usage && d.latency_ms
        ? Number((d.cpu_usage * d.latency_ms).toFixed(2))
        : 0,
  }))

  return (
    <ChartWrapper
      title="Energy Proxy Score (CPU × Latency)"
      description="Lower is better. Approximates inference energy cost for edge devices."
    >
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={energyData}>
          <XAxis dataKey="model" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="energy_proxy" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
