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

export default function CpuChart({ data }: { data: any[] }) {
  return (
    <ChartWrapper
      title="CPU Utilization During Inference"
      description="Measures actual compute cost per model"
    >
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data}>
          <XAxis dataKey="model" hide />
          <YAxis unit="%" />
          <Tooltip />
          <Bar dataKey="cpu_percent" />
        </BarChart>
      </ResponsiveContainer>
    </ChartWrapper>
  )
}
